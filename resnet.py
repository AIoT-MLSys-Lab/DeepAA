import os
import tensorflow as tf
# ref: https://github.com/gahaalt/resnets-in-tensorflow2/blob/master/Models/Resnets.py
_bn_momentum = 0.9

def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer, bias_regularizer=_regularizer,
                                  kernel_initializer='he_normal', use_bias=True)


def bn_relu(x):
    x = tf.keras.layers.experimental.SyncBatchNormalization(momentum=_bn_momentum)(x)
    return tf.keras.layers.ReLU()(x)


def shortcut(x, filters, stride, mode):
    if x.shape[-1] == filters: # maybe and stride==1
        return x
    elif mode == 'B':
        return regularized_padded_conv(filters, 1, strides=stride)(x)
    elif mode == 'B_original':
        x = regularized_padded_conv(filters, 1, strides=stride)(x)
        return tf.keras.layers.experimental.SyncBatchNormalization(momentum=_bn_momentum)(x)
    elif mode == 'A':
        return tf.pad(tf.keras.layers.MaxPool2D(1, stride)(x) if stride > 1 else x,
                      paddings=[(0, 0), (0, 0), (0, 0), (0, filters - x.shape[-1])])
    else:
        raise KeyError("Parameter shortcut_type not recognized!")


def original_block(x, filters, stride=1, **kwargs):
    c1 = regularized_padded_conv(filters, 3, strides=stride)(x)
    c2 = regularized_padded_conv(filters, 3)(bn_relu(c1))
    c2 = tf.keras.layers.experimental.SyncBatchNormalization(momentum=_bn_momentum)(c2)

    mode = 'B_original' if _shortcut_type == 'B' else _shortcut_type
    x = shortcut(x, filters, stride, mode=mode)
    return tf.keras.layers.ReLU()(x + c2)


def preactivation_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)

    c1 = regularized_padded_conv(filters, 3)(flow)
    if _dropout:
        c1 = tf.keras.layers.Dropout(_dropout)(c1)

    c2 = regularized_padded_conv(filters, 3, strides=stride)(bn_relu(c1))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c2


def bootleneck_block(x, filters, stride=1, preact_block=False):
    flow = bn_relu(x)
    if preact_block:
        x = flow

    c1 = regularized_padded_conv(filters // _bootleneck_width, 1)(flow)
    c2 = regularized_padded_conv(filters // _bootleneck_width, 3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(filters, 1)(bn_relu(c2))
    x = shortcut(x, filters, stride, mode=_shortcut_type)
    return x + c3


def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
    global _preact_shortcuts
    preact_block = True if _preact_shortcuts or block_idx == 0 else False

    x = block_type(x, filters, stride, preact_block=preact_block)
    for i in range(num_blocks - 1):
        x = block_type(x, filters)
    return x


def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
           shortcut_type='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
           dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True,
           final_dense_kernel_initializer=None, final_dense_bias_initializer=None):
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
    _bootleneck_width = bootleneck_width  # used in ResNeXts and bootleneck blocks
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type  # used in blocks
    _cardinality = cardinality  # used in ResNeXts
    _dropout = dropout  # used in Wide ResNets
    _preact_shortcuts = preact_shortcuts

    block_types = {'preactivated': preactivation_block,
                   'bootleneck': bootleneck_block,
                   'original': original_block}

    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)

    if block_type == 'original':
        flow = bn_relu(flow)

    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        flow = group_of_blocks(flow,
                               block_type=selected_block,
                               num_blocks=group_size,
                               block_idx=block_idx,
                               filters=feature,
                               stride=stride)

    if block_type != 'original':
        flow = bn_relu(flow)

    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)

    if final_dense_kernel_initializer is not None:
        assert final_dense_bias_initializer is not None, 'make sure kernel and bias initializer is not None at the same time'
        outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer,
                                        kernel_initializer=final_dense_kernel_initializer,
                                        bias_initializer=final_dense_bias_initializer)(flow)
    else:
        outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def load_weights_func(model, model_name):
    try:
        model.load_weights(os.path.join('saved_models', model_name + '.tf'))
    except tf.errors.NotFoundError:
        print("No weights found for this model!")
    return model


def cifar_resnet20(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, input_shape=None, n_classes=None):
    model = Resnet(input_shape=input_shape, n_classes=n_classes, l2_reg=l2_reg, group_sizes=(3, 3, 3), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet20')
    return model


def cifar_resnet32(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(5, 5, 5), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet32')
    return model


def cifar_resnet44(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(7, 7, 7), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet44')
    return model


def cifar_resnet56(block_type='original', shortcut_type='A', l2_reg=1e-4, load_weights=False, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(9, 9, 9), features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet56')
    return model


def cifar_resnet110(block_type='preactivated', shortcut_type='B', l2_reg=1e-4, load_weights=False, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(18, 18, 18),
                   features=(16, 32, 64),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False)
    if load_weights: model = load_weights_func(model, 'cifar_resnet110')
    return model


def cifar_resnet164(shortcut_type='B', load_weights=False, l2_reg=1e-4, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(18, 18, 18),
                   features=(64, 128, 256),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type='bootleneck', preact_shortcuts=True)
    if load_weights: model = load_weights_func(model, 'cifar_resnet164')
    return model


def cifar_resnet1001(shortcut_type='B', load_weights=False, l2_reg=1e-4, input_shape=None):
    model = Resnet(input_shape=input_shape, n_classes=10, l2_reg=l2_reg, group_sizes=(111, 111, 111),
                   features=(64, 128, 256),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type='bootleneck', preact_shortcuts=True)
    if load_weights: model = load_weights_func(model, 'cifar_resnet1001')
    return model


def cifar_wide_resnet(N, K, block_type='preactivated', shortcut_type='B', dropout=0, l2_reg=2.5e-4, n_classes=None, preact_shortcuts=False, input_shape=None):
    assert (N - 4) % 6 == 0, "N-4 has to be divisible by 6"
    lpb = (N - 4) // 6  # layers per block - since N is total number of convolutional layers in Wide ResNet
    model = Resnet(input_shape=input_shape, n_classes=n_classes, l2_reg=l2_reg, group_sizes=(lpb, lpb, lpb),
                   features=(16 * K, 32 * K, 64 * K),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type=block_type, dropout=dropout, preact_shortcuts=preact_shortcuts)
    return model


def cifar_WRN_16_4(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, input_shape=None):
    model = cifar_wide_resnet(16, 4, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, input_shape=input_shape)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_16_4')
    return model


def cifar_WRN_40_4(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, input_shape=None):
    model = cifar_wide_resnet(40, 4, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, input_shape=input_shape)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_40_4')
    return model


def cifar_WRN_16_8(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, input_shape=None):
    model = cifar_wide_resnet(16, 8, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, input_shape=input_shape)
    if load_weights: model = load_weights_func(model, 'cifar_WRN_16_8')
    return model


def cifar_WRN_28_10(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, n_classes=None, preact_shortcuts=False, input_shape=None):
    model = cifar_wide_resnet(28, 10, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, n_classes = n_classes, preact_shortcuts=preact_shortcuts, input_shape=input_shape)
    return model

def cifar_WRN_28_2(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, n_classes=None, preact_shortcuts=False, input_shape=None):
    model = cifar_wide_resnet(28, 2, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, n_classes = n_classes, preact_shortcuts=preact_shortcuts, input_shape=input_shape)
    return model


def cifar_WRN_40_2(shortcut_type='B', load_weights=False, dropout=0, l2_reg=2.5e-4, n_classes=None, preact_shortcuts=False, input_shape=None):
    model = cifar_wide_resnet(40, 2, 'preactivated', shortcut_type, dropout=dropout, l2_reg=l2_reg, n_classes = n_classes, preact_shortcuts=preact_shortcuts, input_shape=input_shape)
    return model

def cifar_resnext(N, cardinality, width, shortcut_type='B', ):
    assert (N - 3) % 9 == 0, "N-4 has to be divisible by 6"
    lpb = (N - 3) // 9  # layers per block - since N is total number of convolutional layers in Wide ResNet
    model = Resnet(input_shape=(32, 32, 3), n_classes=10, l2_reg=1e-4, group_sizes=(lpb, lpb, lpb),
                   features=(16 * width, 32 * width, 64 * width),
                   strides=(1, 2, 2), first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
                   shortcut_type=shortcut_type,
                   block_type='resnext', cardinality=cardinality, width=width)
    return model


if __name__ == '__main__':
    model = cifar_WRN_28_10(dropout=0, l2_reg=5e-4/2., preact_shortcuts=False, n_classes=10)