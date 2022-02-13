_PARALLEL_BATCH_small, _PARALLEL_BATCH_median, _PARALLEL_BATCH_large = 16, 128, 256 # 64
import os

import sys
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(0)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import multiprocessing
import argparse


from augmentation import get_mid_magnitude
from DeepAA_utils import test_loss, test_accuracy, train_loss, train_accuracy
from DeepAA_utils import get_model, get_dataset, get_augmentation, get_loss_fun, get_optim_net, get_optim_policy
from DeepAA_utils import get_lops_luniq, get_policy, get_img_size
from DeepAA_utils import PrefetchGenerator, save_policy

from tensorflow.keras.utils import Progbar
import matplotlib
matplotlib.use('Agg')
from utils import Logger as myLogger
from utils import repeat


parser = argparse.ArgumentParser()
# pretrain
parser.add_argument('--use_model', default='WRN_28_10', type=str, help='Model used for search')
parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset, e.g., cifar10, imagenet')
parser.add_argument('--n_classes', default=100, type=int, help='Number of classes')
parser.add_argument('--nb_epochs', default=45, type=int, help='Number of epochs for pretrain')
parser.add_argument('--pretrain_size', default=5000, type=int, help='Number of images for pretraining')
parser.add_argument('--l_mags', default=13, type=int, help='Number of magnitudes, should be an odd number')
parser.add_argument('--policy_lr', default=0.025, type=float, help='Policy learning rate')
parser.add_argument('--pretrain_lr', default=0.1, type=float, help='maximum learning rate')
parser.add_argument('--batch_size', default=128, type=int, help='Training batch size')
parser.add_argument('--val_batch_size', default=1024, type=int, help='Validation batch size')
parser.add_argument('--test_batch_size', default=512, type=int, help='Testing batch size')
parser.add_argument('--clip_policy_gradient_norm', default=5.0, type=float, help='clipping the policy gradient by norm')
parser.add_argument('--debug', default=False, action='store_true', help='Debugging')
parser.add_argument('--seed', default=1, type=int, help='Random seed')
parser.add_argument('--policy_bn_training', default=False, action='store_true', help='use batchnorm for policy search, Default to False')
parser.add_argument('--n_policies', default=4, type=int, help='Number of policies')
parser.add_argument('--search_bno', default=256, type=int, help='Search steps for each policy')
parser.add_argument('--repeat_random_ops', default=False, action='store_true', help='repeat random operations (randCrop, randFlip, randCutout')
parser.add_argument('--N_repeat_random', default=1, type=int, help='Number to repeats')
parser.add_argument('--use_pool', default=False, action='store_true', help='Using multiprocessing for augmentation')
parser.add_argument('--chunk_size', default=None, type=int, help='Chunk size for augmentation')
parser.add_argument('--EXP_gT_factor', default=4, type=int, help='Expansion factor for calculating gradient')
parser.add_argument('--EXP_G', default=16, type=int, help='Expansion for Jacobian vector product')
parser.add_argument('--train_same_labels', default=16, type=int, help='Sample data from N randomly selected labels')
parser.add_argument('--mode', default='client', type=str, help='Dummy params')
parser.add_argument('--port', default=38277, type=int, help='Dummy params')


args=parser.parse_args()

if args.use_model in ['resnet50']:
    _PARALLEL_BATCH = _PARALLEL_BATCH_small
elif args.use_model in ['WRN_28_10']:
    _PARALLEL_BATCH = _PARALLEL_BATCH_median
elif args.use_model in ['WRN_40_2']:
    _PARALLEL_BATCH = _PARALLEL_BATCH_large
else:
    raise Exception('Unrecognized model {}'.format(args.use_model))

n_cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=n_cpus) if args.use_pool else None
np.random.seed(int(args.seed))
tf.random.set_seed(int(args.seed))


ops_mid_magnitude = get_mid_magnitude(args.l_mags)
args.l_ops, args.l_uniq = get_lops_luniq(args, ops_mid_magnitude)
args.img_size = get_img_size(args)
train_ds, val_ds, test_ds, search_ds = get_dataset(args)
nb_train_steps = len(train_ds)
augmentation_default, augmentation_search, augmentation_test = get_augmentation(args)
_, test_loss_fun, val_loss_fun = get_loss_fun()
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = get_model(args, args.use_model, args.n_classes)
    checkpoint = tf.train.Checkpoint(model=model)
    train_loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                   reduction=tf.keras.losses.Reduction.NONE)
    optim_net = get_optim_net(args, nb_train_steps)
assert args.train_same_labels % mirrored_strategy.num_replicas_in_sync == 0, "Make sure val_same_labels can be divided by num_replicas_in_sync"

available_policies = np.arange(args.l_uniq, dtype=np.int32)[:, np.newaxis]
print(available_policies)
all_using_policies, all_using_optim_policies = [], []
for k in range(args.n_policies):
    policy_train_ = get_policy(args, op_names=augmentation_search.op_names, ops_mid_magnitude=ops_mid_magnitude, available_policies= available_policies)
    optim_policy_ = get_optim_policy(args.policy_lr)
    all_using_policies.append(policy_train_)
    all_using_optim_policies.append(optim_policy_)


train_ds.on_epoch_end()
train_ds_iter = iter(train_ds)
def get_pretrain_data():
    global train_ds_iter
    try:
        images, labels = next(train_ds_iter)
    except:
        train_ds.on_epoch_end()
        train_ds_iter = iter(train_ds)
        images, labels = next(train_ds_iter)
    bs = len(labels)
    images, _ = augmentation_default(images, labels,
                                      [None]*bs, [None]*bs,
                                      use_post_aug=True, pool=pool)
    return tf.convert_to_tensor(images, dtype=tf.float32), tf.convert_to_tensor(labels, tf.int32)

@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, *args.img_size), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.float32)],
)
def train_step(images_aug, labels, clip_gradient_norm):
    bs = len(images_aug)
    with tf.GradientTape() as tape:
        labels_aug_pred = model(images_aug, training=True)
        loss_aug = tf.reduce_mean(train_loss_fun(labels, labels_aug_pred))
        loss_aug += sum(model.losses)
    grad_net = tape.gradient(loss_aug, model.trainable_variables)
    if clip_gradient_norm > 0:
        grad_net, _ = tf.clip_by_global_norm(grad_net, clip_norm=clip_gradient_norm)
    optim_net.apply_gradients(zip(grad_net, model.trainable_variables))
    del tape
    return loss_aug, labels_aug_pred


def pretrain():
    for epoch in range(args.nb_epochs):
        if epoch == args.nb_epochs+1:
            break
        pbar = Progbar(target=nb_train_steps, interval=20, width=30)
        print('\n')
        for bno in range(nb_train_steps):
            images, labels = get_pretrain_data()
            loss, labels_pred = train_step(images, labels, clip_gradient_norm=5.)
            train_loss(loss)  # only record the last method's loss and accuracy
            train_accuracy(labels, labels_pred)
            pbar.update(bno + 1)

    print('Saving the checkpoint to {}'.format('./results/images/ckpt{}/model_ckpt{}'.format(os.environ['CUDA_VISIBLE_DEVICES'], epoch-1)))
    # FixMe: We need to save and then load the pretrain model, otherwise the pretrained model won't be synchronized across all GPUs
    model.save_weights('./results/images/ckpt{}/model_ckpt{}'.format(os.environ['CUDA_VISIBLE_DEVICES'], args.nb_epochs))
    model.load_weights('./results/images/ckpt{}/model_ckpt{}'.format(os.environ['CUDA_VISIBLE_DEVICES'], args.nb_epochs))


search_summary_writer = tf.summary.create_file_writer('./results/images/logs/cuda{}/search'.format(os.environ['CUDA_VISIBLE_DEVICES']))
graph_summary_writer = tf.summary.create_file_writer('./results/images/logs/cuda{}/graph'.format(os.environ['CUDA_VISIBLE_DEVICES']))
save_folder = './results/images/cuda{}'.format(os.environ['CUDA_VISIBLE_DEVICES'])
save_folder_ckpt = './results/images/ckpt{}'.format(os.environ['CUDA_VISIBLE_DEVICES'])
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
if not os.path.isdir(save_folder_ckpt):
    os.mkdir(save_folder_ckpt)
if __name__ == '__main__':
    sys.stdout = myLogger('./results/images/cuda{}/stdout'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    # pretraining
    if 'imagenet' in args.dataset:
        checkpoint.restore('./pretrained_imagenet/imagenet_resnet50_ckpt')
    else:
        pretrain()
    # disable batch normalization updating
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.experimental.SyncBatchNormalization) or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
gradients_like = tf.nest.map_structure(lambda g: tf.zeros_like(g), model.trainable_variables)

@tf.function(
input_signature=[tf.TensorSpec(shape=(None, *args.img_size), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, ), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.float32),
                     tf.TensorSpec(shape=(None, ), dtype=tf.float32)]
)
def step2_cal_JVP_vStep(images_aug2, labels, weight_1, weights_2):
    if not args.debug:
        print('*'*40 + ' retracing step2_cal_JVP_vStep ' + '*'*40)
    with tf.GradientTape() as tape:
        labels_aug_pred = model(images_aug2, training=False)
        loss_aug = train_loss_fun(labels, labels_aug_pred)
    grad_new = tape.gradient(loss_aug, model.trainable_variables, output_gradients = weights_2 * weight_1)
    del tape
    return grad_new

@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, *args.img_size), dtype=tf.float32),
                     tf.TensorSpec(shape=(None,), dtype=tf.int32),
                     tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32),
                     [tf.TensorSpec.from_tensor(v) for v in tf.nest.flatten(gradients_like)]]
)
def step2_cal_JVP_jvpStep(images_aug2, labels, g_norm_train, g_norm_val, tangents):
    if not args.debug:
        print('*'*40 + ' retracing step2_cal_JVP_jStep ' + '*'*40)
    with tf.autodiff.ForwardAccumulator(primals=model.trainable_variables, tangents=tangents) as acc:
        labels_aug_pred = model(images_aug2, training=False)
        loss_aug = train_loss_fun(labels, labels_aug_pred)
    grad_importance_new = acc.jvp(loss_aug) / (g_norm_train * g_norm_val)
    del acc
    return grad_importance_new

@tf.function
def policy_gradient_stage1(reduce_random_mat,
                             images_aug, labels_aug,
                             images_val, labels_val,
                             weight_1, weights_2):
    reduce_random_mat = tf.squeeze(reduce_random_mat)
    images_aug = tf.squeeze(images_aug)
    labels_aug = tf.squeeze(labels_aug)
    images_val = tf.squeeze(images_val)
    labels_val = tf.squeeze(labels_val)
    weight_1 = tf.squeeze(weight_1)
    weights_2 = tf.squeeze(weights_2)

    bs = _PARALLEL_BATCH
    val_bs = tf.shape(images_val)[0]
    mult = tf.cast(val_bs, dtype=tf.float32)

    def batching(L, bs, k): # Get Batch Range
        start = k * bs
        if start + bs > L:
            end = L
        else:
            end = start + bs
        return start, end

    # 1) Step1: Get gradients of augmented and clean data
    def one_batch_grad(imgs, labs, w1, w2, grad):
        grad_new = step2_cal_JVP_vStep(imgs, labs, w1, w2)
        grad = tf.nest.map_structure(lambda g1, g2: g1+g2, grad, grad_new)
        return grad

    @tf.function
    def cal_grad(imgs, labs, w1, w2):
        L = tf.shape(imgs)[0]
        grad0 = tf.nest.map_structure(lambda g: tf.zeros_like(g), model.trainable_variables)
        grad, _ = tf.while_loop(
            cond = lambda grad_acc, k: tf.cast(k, dtype=tf.int32) < tf.cast(tf.math.ceil(tf.cast(L, dtype=tf.float32)/tf.cast(bs, dtype=tf.float32)), dtype=tf.int32),
            body = lambda grad_acc, k: (one_batch_grad(imgs[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                               labs[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                               w1,
                                                               w2[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                               grad_acc), k+1),
            loop_vars = (grad0, tf.constant(0)),
            back_prop = False,
            parallel_iterations = 1,
        )
        return grad

    grad_val = cal_grad(images_val, labels_val, tf.constant(1.0, dtype=tf.float32), tf.ones(val_bs, dtype=tf.float32)/tf.cast(val_bs, dtype=tf.float32))
    grad_train = cal_grad(images_aug, labels_aug, weight_1 * mult, weights_2)
    grad_train = tf.nest.map_structure(lambda g: g/mult, grad_train) # for numerical stability

    # 2) compute tangents
    g_norm_val = tf.linalg.global_norm(grad_val)
    g_norm_train = tf.linalg.global_norm(grad_train)
    gradV_gradT = sum([tf.reduce_sum(g1*g2) for g1, g2 in zip(grad_val, grad_train)])
    gradV_gradT_gradTrainNorm2 = gradV_gradT/(g_norm_train**2)
    tangents = tf.nest.map_structure(lambda g1, g2: g1 - g2 * gradV_gradT_gradTrainNorm2, grad_val, grad_train)

    # 3) compute JVP
    def one_step_JVP(grad_importance_array, imgs, labs, k):
        grad_importance_ = tf.stop_gradient(
            step2_cal_JVP_jvpStep(imgs, labs, g_norm_train, g_norm_val, tangents)
        )
        grad_importance_array = grad_importance_array.write(tf.cast(k, dtype=tf.int32), grad_importance_)
        return grad_importance_array

    @tf.function
    def run_JVP(imgs, labs):
        L = tf.shape(imgs)[0]
        grad_importance_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, element_shape=[None])
        grad_importance_array, _ = tf.while_loop(
            cond = lambda grad_TA, k: tf.cast(k, dtype=tf.int32) < tf.cast(tf.math.ceil(tf.cast(L, dtype=tf.float32)/tf.cast(bs, dtype=tf.float32)), dtype=tf.int32),
            body = lambda grad_TA, k: (one_step_JVP(grad_TA,
                                                     imgs[batching(L,bs,k)[0]:batching(L,bs,k)[1]],
                                                     labs[batching(L,bs,k)[0]:batching(L,bs,k)[1]],
                                                     k), k+1),
            loop_vars = (grad_importance_array, tf.constant(0)),
            back_prop = False,
            parallel_iterations = 1,
        )
        return grad_importance_array.concat()
    grad_importance = run_JVP(images_aug, labels_aug)

    if args.repeat_random_ops:
        grad_importance = tf.matmul(grad_importance[tf.newaxis], reduce_random_mat, transpose_b=True)[0]

    # 4) compute cosine similarity
    cos_sim = gradV_gradT / (g_norm_train * g_norm_val)
    return cos_sim, grad_importance

@tf.function()
def policy_gradient_stage2(reduce_random_mat, images_aug_s, labels_aug_s, images_aug2, labels, images_val, labels_val, weights_gT, weights_G):
    reduce_random_mat = tf.squeeze(reduce_random_mat)
    images_aug_s = tf.squeeze(images_aug_s)
    labels_aug_s = tf.squeeze(labels_aug_s)
    images_val = tf.squeeze(images_val)
    labels_val = tf.squeeze(labels_val)
    weights_gT = tf.squeeze(weights_gT)

    bs = _PARALLEL_BATCH
    val_bs = tf.shape(images_val)[0]
    mult = 1.0

    def batching(L, bs, k): # Get Batch Range
        start = k * bs
        if start + bs > L:
            end = L
        else:
            end = start + bs
        return start, end

    # 1) Step1: Get gradients of augmented and clean data
    def one_batch_grad(imgs, labs, w1, w2, grad):
        grad_new = step2_cal_JVP_vStep(imgs, labs, w1, w2)
        grad = tf.nest.map_structure(lambda g1, g2: g1+g2, grad, grad_new)
        return grad

    @tf.function
    def cal_grad(imgs, labs, w1, w2):
        L = tf.shape(imgs)[0]
        grad0 = tf.nest.map_structure(lambda g: tf.zeros_like(g), model.trainable_variables)
        grad, _ = tf.while_loop(
            cond = lambda grad_acc, k: tf.cast(k, dtype=tf.int32) < tf.cast(tf.math.ceil(tf.cast(L, dtype=tf.float32)/tf.cast(bs, dtype=tf.float32)), dtype=tf.int32),
            body = lambda grad_acc, k: (one_batch_grad(imgs[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                       labs[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                       w1,
                                                       w2[batching(L, bs, k)[0]:batching(L, bs, k)[1]],
                                                       grad_acc), k+1),
            loop_vars = (grad0, tf.constant(0)),
            back_prop = False,
            parallel_iterations = 1,
        )
        return grad

    grad_val = cal_grad(images_val, labels_val, tf.constant(1.0, dtype=tf.float32), tf.ones(val_bs, dtype=tf.float32)/tf.cast(val_bs, dtype=tf.float32))
    grad_train = cal_grad(images_aug_s, labels_aug_s, tf.constant(mult, dtype=tf.float32), weights_gT)

    grad_train = tf.nest.map_structure(lambda g: g/mult, grad_train) # for numerical stability

    # 2) compute tangents
    g_norm_val = tf.linalg.global_norm(grad_val)
    g_norm_train = tf.linalg.global_norm(grad_train)
    gradV_gradT = sum([tf.reduce_sum(g1*g2) for g1, g2 in zip(grad_val, grad_train)])
    gradV_gradT_gradTrainNorm2 = gradV_gradT/(g_norm_train**2)
    tangents = tf.nest.map_structure(lambda g1, g2: g1 - g2 * gradV_gradT_gradTrainNorm2, grad_val, grad_train)

    # 3) compute JVP
    def one_step_JVP(grad_importance_array, imgs, labs, k):
        grad_importance_ = tf.stop_gradient(
            step2_cal_JVP_jvpStep(imgs, labs, g_norm_train, g_norm_val, tangents)
        )
        grad_importance_array = grad_importance_array.write(tf.cast(k, dtype=tf.int32), grad_importance_)
        return grad_importance_array

    @tf.function
    def run_JVP(imgs, labs):
        L = tf.shape(imgs)[0]
        grad_importance_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, infer_shape=False, element_shape=[None])
        grad_importance_array, _ = tf.while_loop(
            cond = lambda grad_TA, k: tf.cast(k, dtype=tf.int32) < tf.cast(tf.math.ceil(tf.cast(L, dtype=tf.float32)/tf.cast(bs, dtype=tf.float32)), dtype=tf.int32),
            body = lambda grad_TA, k: (one_step_JVP(grad_TA,
                                                    imgs[batching(L,bs,k)[0]:batching(L,bs,k)[1]],
                                                    labs[batching(L,bs,k)[0]:batching(L,bs,k)[1]],
                                                    k), k+1),
            loop_vars = (grad_importance_array, tf.constant(0)),
            back_prop = False,
            parallel_iterations = 1,
        )
        return grad_importance_array.concat()

    aug_n, l_seq, w, h, c = images_aug2.shape
    images_aug2_ = tf.reshape(images_aug2, [aug_n * l_seq, w, h, c])
    labels_ = tf.reshape(labels, [aug_n * l_seq])
    grad_importance = run_JVP(images_aug2_, labels_)
    grad_importance = tf.reshape(grad_importance, [aug_n, l_seq])
    if args.repeat_random_ops:
        grad_importance = tf.matmul(grad_importance, reduce_random_mat, transpose_b=True)

    # 4) compute cosine similarity
    cos_sim = gradV_gradT / (g_norm_train * g_norm_val)
    return cos_sim, grad_importance

@tf.function
def distributed_train_stage1(dist_inputs):
    per_replica_cos_sim, per_replica_grad_importance = mirrored_strategy.run(policy_gradient_stage1, args=(*dist_inputs,))
    return mirrored_strategy.experimental_local_results(per_replica_cos_sim), mirrored_strategy.experimental_local_results(per_replica_grad_importance)

@tf.function
def distributed_train_stage2(dist_inputs):
    per_replica_cos_sim, per_replica_grad_importance = mirrored_strategy.run(policy_gradient_stage2, args=(*dist_inputs,))
    return mirrored_strategy.experimental_local_results(per_replica_cos_sim), mirrored_strategy.experimental_local_results(per_replica_grad_importance)

def train_policy_stage1(stage, images_val_, labels_val_, images_batch, labels_batch):
    search_bs = len(images_val_)
    val_bs = len(images_val_[0])
    assert search_bs == len(images_batch), 'Check dimensions'
    assert len(images_val_) % search_bs == 0, 'Use different validation batch for different search data point'

    EXP = 1 # expansion factor
    images_val_, labels_val_ = augmentation_test(sum(images_val_, []), np.concatenate(labels_val_),
                                                   np.array([[0]]*search_bs*val_bs, dtype=np.int32),
                                                   np.array([[0]]*search_bs*val_bs, dtype=np.float32) / float(args.l_mags - 1),
                                                   use_post_aug=True, pool=pool, chunksize=args.chunk_size)

    images_val_ = np.reshape(images_val_, [search_bs, val_bs, *args.img_size])
    labels_val_ = np.reshape(labels_val_, [search_bs, val_bs])

    images_batch = repeat(images_batch, EXP, axis=0)
    labels_batch = repeat(labels_batch, EXP, axis=0)

    ops_dense, mags_dense, reduce_random_mat, ops_mags_idx, probs, probs_exp = all_using_policies[stage-1].get_dense_aug(None, args.repeat_random_ops)
    if isinstance(images_batch[0], list):
        images_aug_last, labels_aug_last = augmentation_search(repeat(sum(images_batch,[]), len(ops_dense), axis=0), repeat(np.concatenate(labels_batch), len(ops_dense), axis=0),
                                                               np.tile(ops_dense, [search_bs * EXP, 1]), np.tile(mags_dense, [search_bs * EXP, 1]).astype(np.float32)/float(args.l_mags-1),
                                                               use_post_aug=False, pool=pool,
                                                               chunksize=None)


    images_aug_last = np.reshape(images_aug_last, [-1, len(ops_dense), *args.img_size])
    labels_aug_last = np.reshape(labels_aug_last, [-1, len(ops_dense)])

    weights_1 = np.ones(search_bs*EXP, dtype=np.float32)
    weights_2 = probs_exp

    assert search_bs % mirrored_strategy.num_replicas_in_sync == 0, 'Make sure that search_bs is multiples of mirrored_trategy'
    all_local_cos_sim, all_local_grad_importance = [], []
    for used_batch in range(0, search_bs, mirrored_strategy.num_replicas_in_sync):
        get_value_fn = lambda ctx: (
                                    tf.constant(reduce_random_mat, dtype=tf.float32),
                                    tf.convert_to_tensor(images_aug_last[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
                                    tf.convert_to_tensor(labels_aug_last[ctx.replica_id_in_sync_group + used_batch], dtype=tf.int32),
                                    tf.convert_to_tensor(images_val_[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
                                    tf.convert_to_tensor(labels_val_[ctx.replica_id_in_sync_group + used_batch], dtype=tf.int32),
                                    tf.convert_to_tensor(weights_1[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
                                    tf.constant(weights_2, dtype=tf.float32),
                                   )
        dist_values = mirrored_strategy.experimental_distribute_values_from_function(get_value_fn)
        all_local_cos_sim_, all_local_grad_importance_  = distributed_train_stage1(dist_values)
        all_local_cos_sim.extend(all_local_cos_sim_)
        all_local_grad_importance.extend(all_local_grad_importance_)
    grad_importance = tf.stack(all_local_grad_importance, axis=0)
    grad_importance = tf.reduce_mean(grad_importance, axis=0)
    mult_factor = 0.25
    with tf.GradientTape() as tape:
        probs = tf.nn.softmax(all_using_policies[stage-1].logits)
        loss_policy_final = -tf.reduce_sum(grad_importance * probs) * mult_factor
    grad_policy = tape.gradient(loss_policy_final, all_using_policies[stage-1].trainable_variables)
    all_using_optim_policies[stage-1].apply_gradients(zip(grad_policy, all_using_policies[stage-1].trainable_variables))
    del tape

def train_policy_stage2(stage, images_val_, labels_val_, images_batch, labels_batch):
    assert stage >= 2, 'depth starts from 2'
    search_bs = len(images_val_)
    val_bs = len(images_val_[0])
    assert search_bs == len(images_batch), 'Check dimension'
    assert len(images_val_) % search_bs == 0, 'Use different validation batch for different search data point'

    images_val_, labels_val_ = augmentation_test(sum(images_val_, []), np.concatenate(labels_val_),
                                                 np.array([[0]]*search_bs*val_bs, dtype=np.int32),
                                                 np.array([[0]]*search_bs*val_bs, dtype=np.float32) / float(args.l_mags - 1),
                                                 use_post_aug=True, pool=pool, chunksize=args.chunk_size)

    images_val_ = np.reshape(images_val_, [search_bs, val_bs, *args.img_size])
    labels_val_ = np.reshape(labels_val_, [search_bs, val_bs])

    EXP_gT = args.l_uniq * args.EXP_gT_factor # Expansion for calculating gradients
    EXP_G = args.EXP_G                        # Expansion for calculating JVP

    images_batch_EXPgT = repeat(images_batch, EXP_gT, axis=0)
    labels_batch_EXPgT = repeat(labels_batch, EXP_gT, axis=0)

    images_batch_EXPG = repeat(images_batch, EXP_G, axis=0)
    labels_batch_EXPG = repeat(labels_batch, EXP_G, axis=0)

    images_aug_s, labels_aug_s = images_batch_EXPgT, labels_batch_EXPgT
    ops_s, mags_s = [], []
    for k_stage in range(1, stage+1):
        dummy_images = [None] * search_bs * EXP_gT
        assert search_bs * EXP_gT == len(images_aug_s)
        assert len(images_aug_s[0]) == 1
        ops_s_, mags_s_, ops_mags_idx_s, probs_sample = all_using_policies[k_stage-1].sample(dummy_images, dummy_images, None, augNum=1)
        ops_s.append(ops_s_)
        mags_s.append(mags_s_)
    ops_s = np.concatenate(ops_s, axis=1)
    mags_s = np.concatenate(mags_s, axis=1)
    images_aug_s, labels_aug_s = augmentation_search(sum(images_aug_s, []), np.concatenate(labels_aug_s, axis=0),
                                                     ops_s, mags_s.astype(np.float32)/float(args.l_mags-1),
                                                     use_post_aug=False, pool=pool,
                                                     chunksize=None)
    images_aug_s = np.reshape(images_aug_s, [search_bs, EXP_gT, *args.img_size])
    labels_aug_s = np.reshape(labels_aug_s, [search_bs, EXP_gT])


    images_aug_k, labels_aug_k = images_batch_EXPG, labels_batch_EXPG
    ops_k, mags_k = [], []
    for k_stage in range(1, stage):
        dummy_images = [None] * search_bs * EXP_G
        assert search_bs * EXP_G == len(images_aug_k)
        assert len(images_aug_k[0]) == 1
        ops_k_, mags_k_, ops_mags_idx_k, probs_sample = all_using_policies[k_stage-1].sample(dummy_images, dummy_images, None, augNum=1)
        ops_k.append(ops_k_)
        mags_k.append(mags_k_)
    ops_k = np.concatenate(ops_k, axis=1)
    mags_k = np.concatenate(mags_k, axis=1)
    images_aug_k, labels_aug_k = augmentation_search(sum(images_aug_k, []), np.concatenate(labels_aug_k, axis=0),
                                                     ops_k, mags_k.astype(np.float32)/float(args.l_mags-1),
                                                     use_post_aug=False, pool=pool, aug_finish=False, chunksize=args.chunk_size)
    ops_dense, mags_dense, reduce_random_mat, ops_mags_idx, probs, probs_exp = all_using_policies[stage-1].get_dense_aug(None, repeat_random_ops=args.repeat_random_ops)
    images_aug_k, labels_aug_k = augmentation_search(repeat(images_aug_k, len(ops_dense), axis=0), np.repeat(labels_aug_k, len(ops_dense), axis=0),
                                                           np.tile(ops_dense, [search_bs * EXP_G, 1]), np.tile(mags_dense, [search_bs * EXP_G, 1]).astype(np.float32)/float(args.l_mags-1),
                                                           use_post_aug=False, pool=pool,
                                                           chunksize=None)

    images_aug_k = np.reshape(images_aug_k, [search_bs, EXP_G, len(ops_dense), *args.img_size])
    labels_aug_k = np.reshape(labels_aug_k, [search_bs, EXP_G, len(ops_dense)])

    weights_gT = np.ones(EXP_gT, dtype=np.float32) / float(EXP_gT)
    weights_G = np.ones(EXP_G, dtype=np.float32) / float(EXP_G)

    assert search_bs % mirrored_strategy.num_replicas_in_sync == 0, 'Make sure that search_bs is multiples of mirrored_trategy'
    all_local_cos_sim, all_local_grad_importance = [], []
    for used_batch in range(0, search_bs, mirrored_strategy.num_replicas_in_sync):
        get_value_fn = lambda ctx: (
            tf.convert_to_tensor(reduce_random_mat, dtype=tf.float32),
            tf.convert_to_tensor(images_aug_s[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
            tf.convert_to_tensor(labels_aug_s[ctx.replica_id_in_sync_group + used_batch], dtype=tf.int32),
            tf.convert_to_tensor(images_aug_k[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
            tf.convert_to_tensor(labels_aug_k[ctx.replica_id_in_sync_group + used_batch], dtype=tf.int32),
            tf.convert_to_tensor(images_val_[ctx.replica_id_in_sync_group + used_batch], dtype=tf.float32),
            tf.convert_to_tensor(labels_val_[ctx.replica_id_in_sync_group + used_batch], dtype=tf.int32),
            tf.convert_to_tensor(weights_gT, dtype=tf.float32),
            tf.convert_to_tensor(weights_G, dtype=tf.float32),
        )
        dist_values = mirrored_strategy.experimental_distribute_values_from_function(get_value_fn)
        all_local_cos_sim_, all_local_grad_importance_ = distributed_train_stage2(dist_values)
        all_local_cos_sim.extend(all_local_cos_sim_)
        all_local_grad_importance.extend(all_local_grad_importance_)
    grad_importance = tf.stack(all_local_grad_importance, axis=0)
    grad_importance = tf.reduce_mean(grad_importance, axis=1)
    assert grad_importance.shape == [search_bs, args.l_uniq], 'Check dimension'
    grad_importance = tf.reduce_mean(grad_importance.numpy(), axis=0) - tf.math.reduce_std(grad_importance.numpy(), axis=0)
    mult_factor = float(search_bs)

    with tf.GradientTape() as tape:
        probs = tf.nn.softmax(all_using_policies[stage - 1].logits)
        loss_policy_final = -tf.reduce_sum(grad_importance * probs) * mult_factor
    grad_policy = tape.gradient(loss_policy_final, all_using_policies[stage - 1].trainable_variables)
    all_using_optim_policies[stage - 1].apply_gradients(zip(grad_policy, all_using_policies[stage - 1].trainable_variables))
    del tape


def search_policy(search_bno, search_bs=16, val_bs=128):
    data_prefetch_iterator = PrefetchGenerator(search_ds, val_ds, args.n_classes, search_bs, val_bs)

    for stage in range(1, args.n_policies + 1):
        pbar = Progbar(target=search_bno, interval=1, width=30)
        for bno in range(search_bno):
            images_val_, labels_val_, images_batch, labels_batch = data_prefetch_iterator.next()

            if stage == 1:
                train_policy_stage1(stage, images_val_, labels_val_, images_batch, labels_batch)
            elif stage > 1:
                train_policy_stage2(stage, images_val_, labels_val_, images_batch, labels_batch)

            pbar.update(bno + 1)


if __name__ == '__main__':
    search_policy(search_bno=args.search_bno, search_bs=args.train_same_labels, val_bs=64)
    save_policy(args, all_using_policies, augmentation_search)

    pool.close()
    pool.join()