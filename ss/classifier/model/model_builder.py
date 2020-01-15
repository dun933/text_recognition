from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization, Activation, Lambda
from keras.layers import Concatenate
import keras.backend as K

block35_iter = 1
block17_iter = 1
d_ft = 1
def conv2d_bn(x, filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x

def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32 // d_ft, 1)
        branch_1 = conv2d_bn(x, 32 // d_ft, 1)
        branch_1 = conv2d_bn(branch_1, 32 // d_ft, 3)
        branch_2 = conv2d_bn(x, 32 // d_ft, 1)
        branch_2 = conv2d_bn(branch_2, 48 // d_ft, 3)
        branch_2 = conv2d_bn(branch_2, 64 // d_ft, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192 // d_ft, 1)
        branch_1 = conv2d_bn(x, 128 // d_ft, 1)
        branch_1 = conv2d_bn(branch_1, 160 // d_ft, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192 // d_ft, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192 // d_ft, 1)
        branch_1 = conv2d_bn(x, 192 // d_ft, 1)
        branch_1 = conv2d_bn(branch_1, 224 // d_ft, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256 // d_ft, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x

def InceptionResNetV2(x):
    x = conv2d_bn(x, 64 // d_ft, 3)
    x = conv2d_bn(x, 64 // d_ft, 1)
    x = conv2d_bn(x, 80 // d_ft, 3)
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96 // d_ft, 1)
    branch_1 = conv2d_bn(x, 48 // d_ft, 1)
    branch_1 = conv2d_bn(branch_1, 64 // d_ft, 5)
    branch_2 = conv2d_bn(x, 64 // d_ft, 1)
    branch_2 = conv2d_bn(branch_2, 96 // d_ft, 3)
    branch_2 = conv2d_bn(branch_2, 96 // d_ft, 3)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64 // d_ft, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(block35_iter):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 256 // d_ft, 3, strides=2, padding='same')
    branch_1 = conv2d_bn(x, 128 // d_ft, 1)
    branch_1 = conv2d_bn(branch_1, 128 // d_ft, 3)
    branch_1 = conv2d_bn(branch_1, 256 // d_ft, 3, strides=2, padding='same')
    branch_pool = MaxPooling2D(3, strides=2, padding='same')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(block17_iter):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    x = conv2d_bn(x, 1028 // d_ft, 1, name='conv_7b')

    return x