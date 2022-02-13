import os
import tensorflow as tf
# ref: https://github.com/gahaalt/resnets-in-tensorflow2/blob/master/Models/Resnets.py
_bn_momentum = 0.9

def regularized_padded_conv(*args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=_regularizer, bias_regularizer=_regularizer,
                                  kernel_initializer='he_normal', use_bias=False)


def bn_relu(x, gamma_initializer='ones'):
    x = tf.keras.layers.experimental.SyncBatchNormalization(momentum=_bn_momentum, gamma_initializer=gamma_initializer)(x)
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


def bootleneck_block(x, filters, stride=1, preact_block=False): # preact_block==False
    # flow = bn_relu(x)
    # if preact_block:
    #     x = flow
    residual = x
    c1 = regularized_padded_conv(filters // _bootleneck_width, 1)(bn_relu(x))
    c2 = regularized_padded_conv(filters // _bootleneck_width, 3, strides=stride)(bn_relu(c1))
    c3 = regularized_padded_conv(filters, 1)(bn_relu(c2))
    if x.shape[-1] != filters or stride != 1:
        residual = shortcut(x, filters, stride, mode=_shortcut_type)
    return tf.keras.layers.ReLU()(residual + tf.keras.layers.experimental.SyncBatchNormalization(momentum=_bn_momentum, gamma_initializer='zeros')(c3))


def group_of_blocks(x, block_type, num_blocks, filters, stride, block_idx=0):
    global _preact_shortcuts
    preact_block = False

    x = block_type(x, filters, stride, preact_block=preact_block)
    for i in range(num_blocks - 1):
        x = block_type(x, filters)
    return x


def Resnet(input_shape, n_classes, l2_reg=1e-4, group_sizes=(2, 2, 2), features=(16, 32, 64), strides=(1, 2, 2),
           shortcut_type='B', block_type='preactivated', first_conv={"filters": 16, "kernel_size": 3, "strides": 1},
           dropout=0, cardinality=1, bootleneck_width=4, preact_shortcuts=True):
    global _regularizer, _shortcut_type, _preact_projection, _dropout, _cardinality, _bootleneck_width, _preact_shortcuts
    _bootleneck_width = bootleneck_width  # used in ResNeXts and bootleneck blocks
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    _shortcut_type = shortcut_type  # used in blocks
    _cardinality = cardinality  # used in ResNeXts
    _dropout = dropout  # used in Wide ResNets
    _preact_shortcuts = preact_shortcuts

    block_types = {
        # 'preactivated': preactivation_block,
        'bootleneck': bootleneck_block,
        'original': original_block
    }

    selected_block = block_types[block_type]
    inputs = tf.keras.layers.Input(shape=input_shape)
    flow = regularized_padded_conv(**first_conv)(inputs)

    # if block_type == 'original':
    flow = bn_relu(flow)
    flow = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding='same')(flow)

    for block_idx, (group_size, feature, stride) in enumerate(zip(group_sizes, features, strides)):
        flow = group_of_blocks(flow,
                               block_type=selected_block,
                               num_blocks=group_size,
                               block_idx=block_idx,
                               filters=feature,
                               stride=stride)

    # if block_type != 'original':
    #     flow = bn_relu(flow)

    flow = tf.keras.layers.GlobalAveragePooling2D()(flow)

    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer, bias_regularizer=_regularizer, use_bias=True)(flow)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def imagenet_resnet50(block_type='bootleneck', shortcut_type='B_original', l2_reg=0.5e-4, load_weights=False, input_shape=(224,224,3), n_classes=1000):
    bootleneck_width = 4
    model = Resnet(input_shape=input_shape, n_classes=n_classes, l2_reg=l2_reg, group_sizes=(3,4,6,3),
                   features=(64*bootleneck_width, 128*bootleneck_width, 256*bootleneck_width, 512*bootleneck_width),
                   strides=(1, 2, 2, 2), first_conv={"filters": 64, "kernel_size": 7, "strides": 2},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False,
                   bootleneck_width=bootleneck_width)
    return model

def imagenet_resnet50_pretrained(input_shape, n_classes, l2_reg):
    _regularizer = tf.keras.regularizers.l2(l2_reg)
    inputs = tf.keras.layers.Input(shape=input_shape)
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape,
                                                         pooling='avg', weights='imagenet')
    base_model.trainable = False
    x = base_model(inputs, training=False) # do not update batch augmentation
    outputs = tf.keras.layers.Dense(n_classes, kernel_regularizer=_regularizer, bias_regularizer=_regularizer, use_bias=True)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def imagenet_resnet18(block_type='original', shortcut_type='B_original', l2_reg=0.5e-4, load_weights=False, input_shape=(224,224,3), n_classes=1000):
    model = Resnet(input_shape=input_shape, n_classes=n_classes, l2_reg=l2_reg, group_sizes=(2,2,2,2),
                   features=(64, 128, 256, 512),
                   strides=(1, 2, 2, 2), first_conv={"filters": 64, "kernel_size": 7, "strides": 2},
                   shortcut_type=shortcut_type,
                   block_type=block_type, preact_shortcuts=False,
                   bootleneck_width=None)
    return model

def load_weights_func(model, model_name):
    try:
        model.load_weights(os.path.join('saved_models', model_name + '.tf'))
    except tf.errors.NotFoundError:
        print("No weights found for this model!")
    return model


if __name__ == '__main__':
    model = imagenet_resnet50()