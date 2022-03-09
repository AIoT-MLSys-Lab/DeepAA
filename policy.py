import tensorflow as tf
import numpy as np
import math
import json

from tensorflow_probability import distributions as tfd

from resnet import Resnet

CIFAR_MEANS = np.array([0.49139968, 0.48215841, 0.44653091], dtype=np.float32)
CIFAR_STDS = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)

SVHN_MEANS = np.array([0.4379, 0.4440, 0.4729], dtype=np.float32)
SVHN_STDS = np.array([0.1980, 0.2010, 0.1970], dtype=np.float32)

IMAGENET_MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class DA_Policy_logits(tf.keras.Model):
    def __init__(self, l_ops, l_mags, l_uniq, op_names, ops_mid_magnitude,
                 N_repeat_random, available_policies, policy_init='identity'):
        super().__init__()
        self.l_uniq = l_uniq
        self.l_ops = l_ops
        self.l_mags = l_mags
        self.N_repeat_random = N_repeat_random
        self.available_policies = available_policies

        if policy_init == 'uniform':
            init_value = tf.constant([0.0]*len(available_policies), dtype=tf.float32)
        elif policy_init == 'identity':
            init_value = tf.constant([8.0] + [0.0]*(len(available_policies)-1), dtype=tf.float32)
            init_value = init_value - tf.reduce_mean(init_value)
        else:
            raise Exception
        self.logits = tf.Variable(initial_value=init_value, trainable=True)

        self.ops_mid_magnitude = ops_mid_magnitude
        self.unique_policy = self._get_unique_policy(op_names, l_ops, l_mags)
        self.N_random, self.repeat_cfg, self.reduce_random_mat = self._get_repeat_random(op_names, l_ops, l_mags,
                                                                                         l_uniq, N_repeat_random)
        self.act = tf.nn.softmax

    def sample(self, images_orig, images, onehot_ops_mags, augNum):
        bs = len(images_orig)
        probs = self.act(self.logits, axis=-1)
        dist = tfd.Categorical(probs=probs)
        samples_om = dist.sample(augNum*bs).numpy()  # (augNum, bs)

        ops_dense, mags_dense, reduce_random_mat, ops_mags_idx, probs, probs_exp = self.get_dense_aug(images, repeat_random_ops=False)
        ops = ops_dense[samples_om]
        mags = mags_dense[samples_om]
        ops_mags_idx_sample = ops_mags_idx[samples_om]
        probs_sample = probs.numpy()[samples_om]

        return ops, mags, ops_mags_idx_sample, probs_sample

    def probs(self, images_orig, images, onehot_ops_mags, training):
        bs = len(images_orig)
        probs = self.act(self.logits, axis=-1)
        probs = tf.repeat(probs[tf.newaxis], bs, axis=0)
        return probs

    def get_dense_aug(self, images, repeat_random_ops):
        ops_uniq, mags_uniq = self.unique_policy
        ops_dense = np.squeeze(ops_uniq)[self.available_policies]
        mags_dense = np.squeeze(mags_uniq)[self.available_policies]
        ops_mags_idx = self.available_policies
        if repeat_random_ops:
            isRepeat = [np.any(np.array(ops_dense == repeat_op_idx), axis=1) for repeat_op_idx in self.repeat_ops_idx]
            isRepeat = np.stack(isRepeat, axis=1)
            isRepeat = np.any(isRepeat, axis=1)
            nRepeat = [self.N_repeat_random if isrepeat else 1 for isrepeat in isRepeat]

            ops_dense = np.repeat(ops_dense, nRepeat, axis=0)
            mags_dense = np.repeat(mags_dense, nRepeat, axis=0)
            reduce_random_mat = np.eye(len(self.available_policies)) / np.array(nRepeat, dtype=np.float32)
            reduce_random_mat = np.repeat(reduce_random_mat, nRepeat, axis=1)
        else:
            nRepeat = [1] * len(self.available_policies)
            reduce_random_mat = np.eye(len(self.available_policies))

        probs = self.act(self.logits)
        probs_exp = np.repeat(probs/np.array(nRepeat, dtype=np.float32), nRepeat, axis=0)
        return ops_dense, mags_dense, reduce_random_mat, ops_mags_idx, probs, probs_exp

    def _get_unique_policy(self, op_names, l_ops, l_mags):
        names_modified = [op_name.split(':')[0] for op_name in op_names]
        ops_list, mags_list = [], []
        repeat_ops_idx = []
        for k_name, name in enumerate(names_modified):
            if self.ops_mid_magnitude[name] == 'random':
                repeat_ops_idx.append(k_name)
                ops_sub, mags_sub = np.array([[k_name]], dtype=np.int32), np.array([[(l_mags - 1) // 2]], dtype=np.int32)
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name]>=0 and self.ops_mid_magnitude[name]<=l_mags-1:
                ops_sub = k_name * np.ones([l_mags - 1, 1], dtype=np.int32)
                mags_sub = np.array([l for l in range(l_mags) if l != self.ops_mid_magnitude[name]], dtype=np.int32)[:, np.newaxis]
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name]<0: #or self.ops_mid_magnitude[name]>l_mags-1):
                ops_sub = k_name * np.ones([l_mags, 1], dtype=np.int32)
                mags_sub = np.arange(l_mags, dtype=np.int32)[:, np.newaxis]
            elif self.ops_mid_magnitude[name] is None:
                ops_sub, mags_sub = np.array([[k_name]], dtype=np.int32), np.array([[(l_mags - 1) // 2]], dtype=np.int32)
            else:
                raise Exception('Unrecognized middle magnitude')
            ops_list.append(ops_sub)
            mags_list.append(mags_sub)
        ops = np.concatenate(ops_list, axis=0)
        mags = np.concatenate(mags_list, axis=0)
        self.repeat_ops_idx = repeat_ops_idx
        return ops.astype(np.int32), mags.astype(np.int32)

    def _get_repeat_random(self, op_names, l_ops, l_mags, l_uniq, N_repeat_random):
        names_modified = [op_name.split(':')[0] for op_name in op_names]
        N_random = sum([1 for name in names_modified if self.ops_mid_magnitude[name]=='random'])
        repeat_cfg = []
        for k_name, name in enumerate(names_modified):
            if self.ops_mid_magnitude[name] == 'random':
                repeat_cfg.append(N_repeat_random) # we may repeat random operations for N_repeat_random times
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name] == -1:
                repeat_cfg.append([1]*l_mags)
            elif self.ops_mid_magnitude[name] is not None and self.ops_mid_magnitude[name] >= 0 and self.ops_mid_magnitude[name]<=l_mags-1:
                repeat_cfg.extend([1]*(l_mags-1))
            elif self.ops_mid_magnitude[name] is None:
                repeat_cfg.append(1)
            else:
                raise Exception
        repeat_cfg = np.array(repeat_cfg, dtype=np.int32)

        reduce_mat = np.eye(l_uniq)/repeat_cfg[np.newaxis].astype(np.float)
        reduce_mat = np.repeat(reduce_mat, repeat_cfg, axis=1)
        return N_random, repeat_cfg, reduce_mat

    @property
    def idx_removed_redundant(self):
        idx_removed_redundant = np.concatenate([[1] if rep == 1 else [1]+[0]*(rep-1) for rep in self.repeat_cfg ]).nonzero()[0]
        assert len(idx_removed_redundant) == self.l_uniq, 'removing the repeated random operations'
        return idx_removed_redundant