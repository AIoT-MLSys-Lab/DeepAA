import os
import logging
import numpy as np
import copy
import random
import datetime

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)


from data_generator import DataGenerator, DataAugmentation
from utils import CTLHistory
from lr_scheduler import GradualWarmup_Cosine_Scheduler
import resnet
from resnet_imagenet import imagenet_resnet50

from data_generator import get_cifar10_data, get_cifar100_data

from augmentation import AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Brightness, Sharpness, \
    Identity, Color, ShearX, ShearY, TranslateX, TranslateY, Rotate
from augmentation import RandCrop, RandCutout, RandFlip, RandCutout60
from augmentation import RandResizeCrop_imagenet, centerCrop_imagenet


from policy import DA_Policy_logits
from augmentation import IMAGENET_SIZE

import torch
import threading
import queue
from imagenet_data_utils import get_imagenet_split

def aug_op_cifar_list():  # oeprators and their ranges
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, -0.3, 0.3),  # 1
        (ShearY, -0.3, 0.3),  # 2
        (TranslateX, -0.45, 0.45),  # 3
        (TranslateY, -0.45, 0.45),  # 4
        (Rotate, -30., 30.),  # 5
        (AutoContrast, 0., 1.),  # 6
        (Invert, 0., 1.),  # 7
        (Equalize, 0., 1.),  # 8
        (Solarize, 0., 256.),  # 9
        (Posterize, 4., 8.),  # 10,
        (Contrast, 0.1, 1.9),  # 11
        (Color, 0.1, 1.9),  # 12
        (Brightness, 0.1, 1.9),  # 13
        (Sharpness, 0.1, 1.9),  # 14
        (RandFlip, 0., 1.0), # 15
        (RandCutout, 0., 1.0), # 16
        (RandCrop, 0., 1.0), # 17
    ]
    names = []
    for op in l:
        info = op.__str__().split(' ')
        name = '{}:({},{}'.format(info[1], info[-2], info[-1])
        names.append(name)

    return l, names

def aug_op_imagenet_list():  # 16 oeprations and their ranges
    l = [
        (Identity, 0., 1.0), # 0
        (ShearX, -0.3, 0.3),  # 1
        (ShearY, -0.3, 0.3),  # 2
        (TranslateX, -0.45, 0.45),  # 3
        (TranslateY, -0.45, 0.45),  # 4
        (Rotate, -30., 30.),  # 5
        (AutoContrast, 0., 1.),  # 6
        (Invert, 0., 1.),  # 7
        (Equalize, 0., 1.),  # 8
        (Solarize, 0., 256.),  # 9
        (Posterize, 4., 8.),  # 10
        (Contrast, 0.1, 1.9),  # 11
        (Color, 0.1, 1.9),  # 12
        (Brightness, 0.1, 1.9),  # 13
        (Sharpness, 0.1, 1.9),  # 14
        (RandFlip, 0., 1.0), # 15
        (RandCutout60, 0., 1.0), # 16
        (RandResizeCrop_imagenet, 0., 1.),
    ]
    names = []
    for op in l:
        info = op.__str__().split(' ')
        name = '{}:({},{}'.format(info[1], info[-2], info[-1])
        names.append(name)

    return l, names


# Get the model
def get_model(args, model, n_classes):
    if model == 'WRN_28_10':
        model = resnet.cifar_WRN_28_10(dropout=0, l2_reg=0.00025,
                                       preact_shortcuts=False, n_classes=n_classes, input_shape=args.img_size)
    elif model == 'WRN_40_2':
        model = resnet.cifar_WRN_40_2(dropout=0, l2_reg=0.00025,
                                      preact_shortcuts=False, n_classes=n_classes, input_shape=args.img_size)
    elif model == 'resnet50':
        model = imagenet_resnet50()
    else:
        raise Exception('Unrecognized model')
    return model

# metric to keep track of
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
train_loss = tf.keras.metrics.Mean()
test_loss = tf.keras.metrics.Mean()

def get_img_size(args):
    if 'cifar' in args.dataset:
        return (32, 32, 3)
    elif 'imagenet' in args.dataset:
        return (*IMAGENET_SIZE, 3)
    else:
        raise Exception

# get the data
def get_dataset(args):
    print('Loading train and retrain dataset.')
    if args.dataset in ['cifar10', 'cifar100']:
        if args.dataset == 'cifar10':
            assert args.n_classes == 10
            x_train_, y_train_, x_val, y_val, x_test, y_test = get_cifar10_data(val_size=10000)
            x_train, y_train = x_train_[:args.pretrain_size], y_train_[:args.pretrain_size]
            x_search, y_search = x_train_[args.pretrain_size:], y_train_[args.pretrain_size:]
        elif args.dataset == 'cifar100':
            assert args.n_classes == 100
            x_train_, y_train_, x_val, y_val, x_test, y_test = get_cifar100_data(val_size=10000)
            x_train, y_train = x_train_[:args.pretrain_size], y_train_[:args.pretrain_size]
            x_search, y_search = x_train_[args.pretrain_size:], y_train_[args.pretrain_size:]
        train_ds = DataGenerator(x_train, y_train, batch_size=args.batch_size, drop_last=True)
        search_ds = DataGenerator(x_search, y_search, batch_size=args.batch_size, drop_last=True)
        val_ds = DataGenerator(x_val, y_val, batch_size=args.val_batch_size, drop_last=True)
        test_ds = DataGenerator(x_test, y_test, batch_size=args.test_batch_size, drop_last=False, shuffle=False)  # setting shuffle=False for parallel evaluation
    elif args.dataset == 'imagenet':
        assert args.n_classes == 1000
        def collate_fn_imagenet_list(l):  # return a list
            images, labels = zip(*l)
            assert images[0].dtype == np.uint8
            return list(images), np.array(labels, dtype=np.int32)
        if args.dataset == 'imagenet':
            train_ds_total, val_ds, search_ds, train_ds, test_ds = get_imagenet_split(n_GPU=1, seed=300)
        assert len(train_ds) == 1 and isinstance(train_ds, list), 'Train_ds should be a length=1 list'
        train_ds = train_ds[0]
        test_ds = torch.utils.data.DataLoader(
            test_ds, batch_size=256, shuffle=False, num_workers=64,
            pin_memory=False,
            drop_last=False, sampler=None,
            collate_fn=collate_fn_imagenet_list,
        )
    else:
        raise Exception('Unrecognized dataset')

    return train_ds, val_ds, test_ds, search_ds

def get_augmentation(args):
    if 'cifar' in args.dataset:
        augmentation_default = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset, image_shape=args.img_size,
                                        ops_list=(None, None),
                                        default_pre_aug=None,
                                        default_post_aug=[RandCrop,
                                                          RandFlip,
                                                          RandCutout])

        augmentation_search = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset, image_shape=args.img_size,
                                                ops_list=aug_op_cifar_list(),
                                                default_pre_aug=None,
                                                default_post_aug=None)

        augmentation_test = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset, image_shape=args.img_size,
                                             ops_list=(None, None),
                                             default_pre_aug=None,
                                             default_post_aug=None)
    elif 'imagenet' in args.dataset:
        augmentation_default = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset,
                                                image_shape=args.img_size,
                                                ops_list=(None, None),
                                                default_pre_aug=None,
                                                default_post_aug=[RandResizeCrop_imagenet, #
                                                                  RandFlip])

        augmentation_search = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset, image_shape=args.img_size,
                                               ops_list=aug_op_imagenet_list(),
                                               default_pre_aug=None,
                                               default_post_aug=None)


        augmentation_test = DataAugmentation(num_classes=args.n_classes, dataset=args.dataset,
                                                image_shape=args.img_size,
                                                ops_list=(None, None),
                                                default_pre_aug=None,
                                                default_post_aug=[
                                                    centerCrop_imagenet,
                                                ])
    return augmentation_default, augmentation_search, augmentation_test

def get_optim_net(args, nb_train_steps):
    scheduler_lr = GradualWarmup_Cosine_Scheduler(starting_lr=0., initial_lr=args.pretrain_lr,
                                                  ending_lr=1e-7,
                                                  warmup_steps= 0,
                                                  total_steps=nb_train_steps * args.nb_epochs)

    optim_net = tf.optimizers.SGD(learning_rate=scheduler_lr, momentum=0.9, nesterov=True)
    return optim_net




def get_policy(args, op_names, ops_mid_magnitude, available_policies):
    policy = DA_Policy_logits(args.l_ops, args.l_mags, args.l_uniq,
                              op_names=op_names,
                              ops_mid_magnitude=ops_mid_magnitude, N_repeat_random=args.N_repeat_random,
                              available_policies=available_policies)
    return policy

def get_optim_policy(policy_lr):
    optim_policy = tf.optimizers.Adam(learning_rate=policy_lr, beta_1=0.9, beta_2=0.999)
    return optim_policy


# get the loss
def get_loss_fun():
    train_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                   reduction=tf.keras.losses.Reduction.NONE)
    test_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                  reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    val_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    return train_loss_fun, test_loss_fun, val_loss_fun


def get_lops_luniq(args, ops_mid_magnitude):
    if 'cifar' in args.dataset:
        _, op_names = aug_op_cifar_list()
    elif 'imagenet' in args.dataset:
        _, op_names = aug_op_imagenet_list()
    else:
        raise Exception('Unknown dataset ={}'.format(args.dataset))

    names_modified = [op_name.split(':')[0] for op_name in op_names]
    l_ops = len(op_names)
    l_uniq = 0
    for k_name, name in enumerate(names_modified):
        mid_mag = ops_mid_magnitude[name]
        if mid_mag == 'random':
           l_uniq += 1 # The op is a random op, however we only sample one op
        elif mid_mag is not None and mid_mag >=0 and mid_mag <= args.l_mags-1:
            l_uniq += args.l_mags-1
        elif mid_mag is not None and mid_mag == -1: # magnitude==-1 means all l_mags are independnt policies; or mid_mag > args.l_mags-1)
            l_uniq += args.l_mags
        elif mid_mag is None:
            l_uniq += 1
        else:
            raise Exception('mid_mag = {} is invalid'.format(mid_mag))
    return l_ops, l_uniq

def get_all_policy(policy_train):
    l_ops, l_mags = policy_train.l_ops, policy_train.l_mags
    ops, mags = np.meshgrid(np.arange(l_ops), np.arange(l_mags), indexing='ij')
    ops = np.reshape(ops, [l_ops*l_mags,1])
    mags = np.reshape(mags, [l_ops*l_mags,1])
    return ops.astype(np.int32), mags.astype(np.int32)

class PrefetchGenerator(threading.Thread):
    def __init__(self, search_ds, val_ds, n_classes, search_bs=8, val_bs=64):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(1)
        self.search_ds = search_ds
        self.val_ds = val_ds
        self.n_classes = n_classes
        self.search_bs = search_bs
        self.val_bs = val_bs
        self.daemon = True
        self.start()

    @staticmethod
    def sample_label_and_batch(dataset, bs, n_classes, MAX_iterations=100):
        for k in range(MAX_iterations):
            try:
                lab = random.randint(0, n_classes-1)
                imgs, labs = dataset.sample_labeled_data_batch(lab, bs)
            except:
                print('Insufficient data in a single class, try {}/{}'.format(k, MAX_iterations))
                continue
            return lab, imgs, labs
        raise Exception('Maximum number of iteration {} reached'.format(MAX_iterations))

    def run(self):
        while True:
            images_val, labels_val, images_train, labels_train = [], [], [], []
            for _ in range(self.search_bs):
                lab, imgs_val, labs_val = PrefetchGenerator.sample_label_and_batch(self.val_ds, self.val_bs, self.n_classes)
                imgs_train, labs_train = self.search_ds.sample_labeled_data_batch(lab, 1)
                images_val.append(imgs_val)
                labels_val.append(labs_val)
                images_train.append(imgs_train)
                labels_train.append(labs_train)
            self.queue.put( (images_val, labels_val, images_train, labels_train) )

    def next(self):
        next_item = self.queue.get()
        return next_item


def save_policy(args, all_using_policies, augmentation_search):
    ops, mags = all_using_policies[0].unique_policy
    op_names = augmentation_search.op_names
    policy_probs = []
    for k_policy, policy in enumerate(all_using_policies):
        policy_probs.append(tf.nn.softmax(policy.logits).numpy())
    policy_probs = np.stack(policy_probs, axis=0)

    np.savez('./policy_port/policy_DeepAA_{}.npz'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")),
             policy_probs=policy_probs, l_ops=args.l_ops, l_mags=args.l_mags,
             ops=ops, mags=mags, op_names=op_names)