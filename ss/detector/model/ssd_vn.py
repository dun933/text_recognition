"""Keras implementation of SSD."""

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Concatenate, Multiply, Add
from keras.layers import Input, Reshape, BatchNormalization, Activation, ZeroPadding2D, Permute
from keras.layers import merge
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras import initializers, regularizers

from model.ssd_layers import Normalize, PriorBox
from model.coord_layer import CoordinateChannel2D

# from model.coord_layer import CoordinateChannel2D


def SSD300(input_shape, num_classes=21):
    """SSD300 architecture.

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """

    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    # net['input'] = input_tensor

    conv1_1 = Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=input_shape,
                     name='conv1_1')(input_tensor)
    conv1_2 = Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1_2')(conv1_1)

    pool1 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='pool1')(conv1_2)
    # Block 2
    conv2_1 = Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='pool2')(conv2_2)
    # Block 3
    conv3_1 = Conv2D(256, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='pool3')(conv3_3)
    # Block 4
    conv4_1 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         padding='same',
                         name='pool4')(conv4_3)
    # Block 5
    conv5_1 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3),
                         strides=(1, 1),
                         padding='same',
                         name='pool5')(conv5_3)
    # FC6
    fc6 = Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6),
                 activation='relu',
                 padding='same',
                 name='fc6')(pool5)
    # x = Dropout(0.5, name='drop6')(x)
    # FC7
    fc7 = Conv2D(1024, kernel_size=(1, 1),
                 activation='relu',
                 padding='same',
                 name='fc7')(fc6)
    # x = Dropout(0.5, name='drop7')(x)
    # Block 6
    conv6_1 = Conv2D(256, kernel_size=(1, 1),
                     activation='relu',
                     padding='same',
                     name='conv6_1')(fc7)
    # conv6_padding = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu',
                     padding='same',
                     name='conv6_2')(conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, kernel_size=(1, 1),
                     activation='relu',
                     padding='same',
                     name='conv7_1')(conv6_2)
    conv7_padding = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu',
                     padding='valid',
                     name='conv7_2')(conv7_padding)
    # Block 8
    conv8_1 = Conv2D(128, kernel_size=(1, 1),
                     activation='relu',
                     padding='same',
                     name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, kernel_size=(3, 3),
                     strides=(2, 2),
                     activation='relu',
                     padding='same',
                     name='conv8_2')(conv8_1)

    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    x = Conv2D(num_priors * 4, kernel_size=(3, 3),
               padding='same',
               name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    conv4_3_norm_mbox_loc_flat = flatten(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3),
               padding='same',
               name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    conv4_3_norm_mbox_conf_flat = flatten(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[0.5, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # Prediction from fc7
    num_priors = 6
    fc7_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3),
                          padding='same',
                          name='fc7_mbox_loc')(fc7)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    fc7_mbox_loc_flat = flatten(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3),
                           padding='same',
                           name=name)(fc7)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    fc7_mbox_conf_flat = flatten(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)

    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3),
               padding='same',
               name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    conv6_2_mbox_loc_flat = flatten(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3),
               padding='same',
               name=name)(conv6_2)
    conv6_2_mbox_conf = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    conv6_2_mbox_conf_flat = flatten(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)

    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3),
               padding='same',
               name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    conv7_2_mbox_loc_flat = flatten(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3),
               padding='same',
               name=name)(conv7_2)
    conv7_2_mbox_conf = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    conv7_2_mbox_conf_flat = flatten(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)

    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3),
               padding='same',
               name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    conv8_2_mbox_loc_flat = flatten(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3),
               padding='same',
               name=name)(conv8_2)
    conv8_2_mbox_conf = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    conv8_2_mbox_conf_flat = flatten(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)

    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_loc_flat = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(pool6)
    pool6_mbox_conf_flat = x
    priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape,
                             name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)

    # Gather all predictions
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_flat,
                                                     fc7_mbox_loc_flat,
                                                     conv6_2_mbox_loc_flat,
                                                     conv7_2_mbox_loc_flat,
                                                     conv8_2_mbox_loc_flat,
                                                     pool6_mbox_loc_flat])

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_flat,
                                                       fc7_mbox_conf_flat,
                                                       conv6_2_mbox_conf_flat,
                                                       conv7_2_mbox_conf_flat,
                                                       conv8_2_mbox_conf_flat,
                                                       pool6_mbox_conf_flat])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               pool6_mbox_priorbox])

    '''
    print('===================')
    print(mbox_loc._keras_shape)
    print(conv4_3_norm_mbox_loc_flat._keras_shape)
    print(fc7_mbox_loc_flat._keras_shape)
    print(conv6_2_mbox_loc_flat._keras_shape)
    print(conv7_2_mbox_loc_flat._keras_shape)
    print(conv8_2_mbox_loc_flat._keras_shape)
    print(pool6_mbox_loc_flat._keras_shape)
    print('===================')
    print(mbox_conf._keras_shape)
    print(conv4_3_norm_mbox_conf_flat._keras_shape)
    print(fc7_mbox_conf_flat._keras_shape)
    print(conv6_2_mbox_conf_flat._keras_shape)
    print(conv7_2_mbox_conf_flat._keras_shape)
    print(conv8_2_mbox_conf_flat._keras_shape)
    print(pool6_mbox_conf_flat._keras_shape)
    print('===================')
    print(mbox_priorbox._keras_shape)
    print(conv4_3_norm_mbox_priorbox._keras_shape)
    print(fc7_mbox_priorbox._keras_shape)
    print(conv6_2_mbox_priorbox._keras_shape)
    print(conv7_2_mbox_priorbox._keras_shape)
    print(conv8_2_mbox_priorbox._keras_shape)
    print(pool6_mbox_priorbox._keras_shape)
    '''

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                                                           mbox_conf,
                                                           mbox_priorbox])

    model = Model(input_tensor, predictions)
    return model


def SSD_AICR(input_shape, num_classes=21):
    """SSD300 architecture - Modified Reference : Feature-Fused SSD: Fast Detection for Small Objects

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325 SSD300
        https://arxiv.org/pdf/1709.05054 Feature-Fused SSD: Fast Detection for Small Objects
    """
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # Block 1
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1_1)
    conv1_3 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block1_conv3')(conv1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_1')(conv1_2)
    block1 = Concatenate(axis=-1, name='merge_conv_block_1')([conv1_3, pool1])
    block1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv4')(block1)
    block1 = BatchNormalization(name='bn_block_1')(block1)
    block1 = Activation('relu', name='relu_block_1')(block1)

    # Block 2
    conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1)
    conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2_1)
    conv2_3 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block2_conv3')(conv2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_1')(conv2_2)
    block2 = Concatenate(axis=-1, name='merge_conv_block_2')([conv2_3, pool2])
    block2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv4')(block2)
    block2 = BatchNormalization(name='bn_block_2')(block2)
    block2 = Activation('relu', name='relu_block_2')(block2)

    # Block 3
    conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2)
    conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block3_conv4')(conv3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_1')(conv3_3)
    block3 = Concatenate(axis=-1, name='merge_conv_block_3')([conv3_4, pool3])
    block3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv5')(block3)
    block3 = BatchNormalization(name='bn_block_3')(block3)
    block3 = Activation('relu', name='relu_block_3')(block3)

    # Block 4
    conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3)
    conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block4_conv4')(conv4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_1')(conv4_3)
    block4 = Concatenate(axis=-1, name='merge_conv_block_4')([conv4_4, pool4])
    block4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv5')(block4)
    block4 = BatchNormalization(name='bn_block_4')(block4)
    block4 = Activation('relu', name='relu_block_4')(block4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='block5_conv4')(conv5_3)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='block5_pool_1')(conv5_3)
    block5 = Concatenate(axis=-1, name='merge_conv_block_5')([conv5_4, pool5])
    block5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv5')(block5)
    block5 = BatchNormalization(name='bn_block_5')(block5)
    block5 = Activation('relu', name='relu_block_5')(block5)

    # FC6
    fc6 = Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(block5)
    # FC7
    fc7 = Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same', name='fc7')(fc6)
    # Block 6
    conv6_1 = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(
        conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_padding = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(
        conv7_padding)
    # Block 8
    conv8_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(
        conv8_1)

    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # prediction from conv3_3
    conv3_3_norm = Normalize(20, name='conv3_3_norm')(conv3_3)
    num_priors = 2
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv3_3_norm_mbox_loc')(conv3_3_norm)
    conv3_3_norm_mbox_loc = x
    flatten = Flatten(name='conv3_3_norm_mbox_loc_flat')
    conv3_3_norm_mbox_loc_flat = flatten(conv3_3_norm_mbox_loc)
    name = 'conv3_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv3_3_norm)
    conv3_3_norm_mbox_conf = x
    flatten = Flatten(name='conv3_3_norm_mbox_conf_flat')
    conv3_3_norm_mbox_conf_flat = flatten(conv3_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 20.0, aspect_ratios=[0.5], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv3_3_norm_mbox_priorbox')
    conv3_3_norm_mbox_priorbox = priorbox(conv3_3_norm)

    # Prediction from conv4_3
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    conv4_3_norm_mbox_loc_flat = flatten(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    conv4_3_norm_mbox_conf_flat = flatten(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # Prediction from fc7
    num_priors = 5
    fc7_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    fc7_mbox_loc_flat = flatten(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(fc7)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    fc7_mbox_conf_flat = flatten(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 40.0, max_size=70.0, aspect_ratios=[0.7, 0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)

    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    conv6_2_mbox_loc_flat = flatten(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv6_2)
    conv6_2_mbox_conf = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    conv6_2_mbox_conf_flat = flatten(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 70.0, max_size=120.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)

    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    conv7_2_mbox_loc_flat = flatten(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv7_2)
    conv7_2_mbox_conf = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    conv7_2_mbox_conf_flat = flatten(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 110.0, max_size=160.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)

    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    conv8_2_mbox_loc_flat = flatten(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv8_2)
    conv8_2_mbox_conf = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    conv8_2_mbox_conf_flat = flatten(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 160.0, max_size=210.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)

    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_loc_flat = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(pool6)
    pool6_mbox_conf_flat = x
    priorbox = PriorBox(img_size, 210.0, max_size=320.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape,
                             name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)

    # Gather all predictions
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv3_3_norm_mbox_loc_flat,
                                                     conv4_3_norm_mbox_loc_flat,
                                                     fc7_mbox_loc_flat,
                                                     conv6_2_mbox_loc_flat,
                                                     conv7_2_mbox_loc_flat,
                                                     conv8_2_mbox_loc_flat,
                                                     pool6_mbox_loc_flat])

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv3_3_norm_mbox_conf_flat,
                                                       conv4_3_norm_mbox_conf_flat,
                                                       fc7_mbox_conf_flat,
                                                       conv6_2_mbox_conf_flat,
                                                       conv7_2_mbox_conf_flat,
                                                       conv8_2_mbox_conf_flat,
                                                       pool6_mbox_conf_flat])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv3_3_norm_mbox_priorbox,
                                                               conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               pool6_mbox_priorbox])

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                                                           mbox_conf,
                                                           mbox_priorbox])

    model = Model(input_tensor, predictions)
    return model


def _conv_block_bn(net, filters, kernel_size, padding='same', strides=(1, 1), separble=False):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    if separble:
        net = SeparableConv2D(filters, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(
            net)
    else:
        net = Conv2D(filters, kernel_size, padding=padding, strides=strides,
                     kernel_regularizer=regularizers.l2(0.00004),
                     kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal',
                                                                     seed=None))(net)

    net = BatchNormalization(axis=channel_axis)(net)
    net = Activation('relu')(net)

    return net


def _conv_block_up(net, filters, kernel_size, padding='same', strides=(2, 2)):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal')(net)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def _block_inception_a(input):  # 256,  40
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = _conv_block_bn(input, 96, (1, 1))  # 96,  40

    branch_1 = _conv_block_bn(input, 64, (1, 1))  # 64,  40
    branch_1 = _conv_block_bn(branch_1, 96, (3, 3))  # 96,  40

    branch_2 = _conv_block_bn(input, 64, (1, 1))  # 64,  40
    branch_2 = _conv_block_bn(branch_2, 96, (3, 3))  # 96,  40
    branch_2 = _conv_block_bn(branch_2, 96, (3, 3))  # 96,  40

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = _conv_block_bn(branch_3, 96, (1, 1))  # 96,  40

    x = Concatenate(axis=channel_axis)([branch_0, branch_1, branch_2, branch_3])  # 512,  40

    return x  # 512,  40


def _block_inception_b(input):  # 512,  20
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = _conv_block_bn(input, 384, (1, 1))  # 384,  20

    branch_1 = _conv_block_bn(input, 192, (1, 1))  # 192,  20
    branch_1 = _conv_block_bn(branch_1, 224, (5, 1))  # 224,  20
    branch_1 = _conv_block_bn(branch_1, 256, (1, 5))  # 256,  20

    branch_2 = _conv_block_bn(input, 192, (1, 1))  # 192,  20
    branch_2 = _conv_block_bn(branch_2, 192, (5, 1))  # 192,  20
    branch_2 = _conv_block_bn(branch_2, 224, (1, 5))  # 224,  20
    branch_2 = _conv_block_bn(branch_2, 224, (5, 1))  # 224,  20
    branch_2 = _conv_block_bn(branch_2, 256, (1, 5))  # 256,  20

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = _conv_block_bn(branch_3, 128, (1, 1))  # 128,  20

    x = Concatenate(axis=channel_axis)([branch_0, branch_1, branch_2, branch_3])

    return x  # 1024, 20


def _block_inception_c(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = _conv_block_bn(input, 256, (1, 1))

    branch_1 = _conv_block_bn(input, 384, (1, 1))
    branch_10 = _conv_block_bn(branch_1, 256, (1, 3))
    branch_11 = _conv_block_bn(branch_1, 256, (3, 1))
    branch_1 = Concatenate(axis=channel_axis)([branch_10, branch_11])  # 512

    branch_2 = _conv_block_bn(input, 384, (1, 1))
    branch_2 = _conv_block_bn(branch_2, 448, (3, 1))
    branch_2 = _conv_block_bn(branch_2, 512, (1, 3))
    branch_20 = _conv_block_bn(branch_2, 256, (1, 3))
    branch_21 = _conv_block_bn(branch_2, 256, (3, 1))
    branch_2 = Concatenate(axis=channel_axis)([branch_20, branch_21])  # 512

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = _conv_block_bn(branch_3, 256, (1, 1))

    x = Concatenate(axis=channel_axis)([branch_0, branch_1, branch_2, branch_3])  # 1536
    return x


def _block_reduction_a(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = _conv_block_bn(input, 384, (3, 3), strides=(2, 2), separble=False)

    branch_1 = _conv_block_bn(input, 192, (1, 1))
    branch_1 = _conv_block_bn(branch_1, 224, (3, 3), separble=False)
    branch_1 = _conv_block_bn(branch_1, 256, (3, 3), strides=(2, 2), separble=False)

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)  # 256,   20

    x = Concatenate(axis=channel_axis)([branch_0, branch_1, branch_2])
    return x


def _block_reduction_b(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = _conv_block_bn(input, 192, (1, 1))
    branch_0 = _conv_block_bn(branch_0, 192, (3, 3), strides=(2, 2), separble=False)  # 192,   10

    branch_1 = _conv_block_bn(input, 256, (1, 1))  # 256,   20
    branch_1 = _conv_block_bn(branch_1, 256, (1, 5))  # 256,   20
    branch_1 = _conv_block_bn(branch_1, 320, (5, 1))  # 320,   20
    branch_1 = _conv_block_bn(branch_1, 320, (3, 3), strides=(2, 2), separble=False)  # 320,   10

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(input)  # 1024,   10

    x = Concatenate(axis=channel_axis)([branch_0, branch_1, branch_2])  # 1536,   10
    return x


def _block_stem(input):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    net = CoordinateChannel2D()(input)

    # net = _conv_block_bn(net, 32, (3, 3))               #  32,  160 → 하나의 Convolution Block을 제거하는 대신 Filter 수를 32개에서 64개로 늘림.
    net = _conv_block_bn(net, 64, (3, 3))  # 32,  160
    net = _conv_block_bn(net, 64, (3, 3), strides=(2, 2))  # 64,  160
    x = _conv_block_bn(net, 96, (3, 3), strides=(2, 2), separble=False)  # 96,   80
    y = MaxPooling2D((2, 2), strides=(2, 2))(net)  # 64,   80
    net = Concatenate(axis=channel_axis)([x, y])  # 160,   80

    x = _conv_block_bn(net, 64, (1, 1))  # 64,   80
    x = _conv_block_bn(x, 96, (3, 3), separble=False)  # 64,   80

    y = _conv_block_bn(net, 64, (1, 1))  # 64,   80
    y = _conv_block_bn(y, 64, (5, 1))  # 64,   80
    y = _conv_block_bn(y, 64, (1, 5))  # 64,   80
    y = _conv_block_bn(y, 96, (3, 3), separble=False)  # 96,   80
    net = Concatenate(axis=channel_axis)([x, y])  # 192,   80
    ## feature extraction - 0 : conv0 추가
    conv0 = _conv_block_bn(net, 64, (3, 3), separble=True)  # 64,   80

    x = _conv_block_bn(conv0, 192, (3, 3), strides=(2, 2), separble=False)  # 192,   40
    y = MaxPooling2D((2, 2), strides=(2, 2))(conv0)  # 128,   40
    net = Concatenate(axis=channel_axis)([x, y])  # 256,   40

    return net, conv0


def squeeze_excite_block(net, ratio=4):
    init = net
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = Multiply()([init, se])
    return x


def SSD_AICR_v01(input_shape, num_classes=21):  # inception v4 style with Seperable Convolutions
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    ## Stem layers + feature extraction conv0 for aicr
    net, conv0 = _block_stem(input_tensor)  # (256,  40), (64, 80)
    conv0 = squeeze_excite_block(conv0)

    for idx in range(2):  # 4
        net = _block_inception_a(net)  # 384,  40
        net = squeeze_excite_block(net)
    ## feature extraction - 1 : conv1
    conv1 = net  # 384,  40

    ## reduction layer
    ## feature extraction - 2 : conv2
    net = _block_reduction_a(net)  # (384, 40)→(1024, 20)

    for idx in range(3):  # 7
        net = _block_inception_b(net)  # 1024,  20
        net = squeeze_excite_block(net)

    conv2 = net  # 1024,  20

    ## reduction layer
    ## feature extraction - 3 : conv3
    net = _block_reduction_b(net)  # (1024, 20)→(1536, 10)
    conv3 = net  # 1536,  10

    ## feature extraction - 4 : conv4
    for idx in range(2):  # 3
        net = _block_inception_c(net)  # 1536,  10
        net = squeeze_excite_block(net)

    conv4 = _conv_block_bn(net, 256, (3, 3), strides=(2, 2), separble=True)  # 512,  5

    net = _conv_block_bn(conv4, 128, (1, 1))
    conv5_1 = _conv_block_bn(net, 128, (3, 3), strides=(1, 1), padding='valid', separble=True)  # 128, 3
    conv5_2 = _conv_block_bn(net, 128, (3, 3), strides=(2, 2), padding='same', separble=True)  # 128, 3
    conv5 = Concatenate(axis=channel_axis)([conv5_1, conv5_2])  # 256, 3
    conv5 = _conv_block_bn(conv5, 256, (3, 3), separble=True)  # 256, 3

    net = _conv_block_bn(conv5, 128, (1, 1))
    conv6 = _conv_block_bn(net, 256, (3, 3), strides=(1, 1), padding='valid', separble=True)  # 256, 1
    conv6 = GlobalAveragePooling2D()(conv6)

    # prediction from conv3_3 : 80x80X 256 - Conv0 + Up(Conv1)
    # Prediction from conv4_3 : 40x40x 512 - Conv1 + Up(Conv2)
    # Prediction from fc7     : 20x20x1024 - Conv2 + Up(Conv3)
    # Prediction from conv6_2 : 10x10x 512 - Conv3 + Up(Conv4)
    # Prediction from conv7_2 :  5x 5x 256 - Conv4 + Up(Conv5)
    # Prediction from conv8_2 :  3x 3x 256 - Conv5 
    # Prediction from pool6   :  1x 1x 256 - Conv6

    # upsampling Block
    net = _conv_block_up(conv5, 64, (3, 3), strides=(1, 1), padding='valid')  # 256,  5
    conv4 = _conv_block_bn(conv4, 64, (3, 3), separble=False)  # 256,  5
    conv4 = Normalize(20, name='conv_up_1_0_normalize')(conv4)
    net = Normalize(10, name='conv_up_1_1_normalize')(net)
    conv4 = Add()([net, conv4])
    conv4 = BatchNormalization(axis=channel_axis)(conv4)
    # conv4 = Activation('relu')(conv4)
    # conv4 = Concatenate(axis=channel_axis)([net, conv4])                        # 512,  5
    # conv4 = _conv_block_bn(conv4, 256, (3, 3), separble=True)                 # 256,  5

    net = _conv_block_up(conv4, 64, (3, 3), strides=(2, 2), padding='same')  # 256, 10
    conv3 = _conv_block_bn(conv3, 64, (3, 3), separble=False)  # 256,  5
    conv3 = Normalize(20, name='conv_up_2_0_normalize')(conv3)
    net = Normalize(10, name='conv_up_2_1_normalize')(net)
    conv3 = Add()([net, conv3])
    conv3 = BatchNormalization(axis=channel_axis)(conv3)
    # conv3 = Activation('relu')(conv3)
    # conv3 = Concatenate(axis=channel_axis)([net, conv3])                        # 512, 10
    # conv3 = _conv_block_bn(conv3, 256, (3, 3), separble=True)                   # 256, 10

    net = _conv_block_up(conv3, 64, (3, 3), strides=(2, 2), padding='same')  # 256, 20
    conv2 = _conv_block_bn(conv2, 64, (3, 3), separble=False)  # 256,  5
    conv2 = Normalize(20, name='conv_up_3_0_normalize')(conv2)
    net = Normalize(10, name='conv_up_3_1_normalize')(net)
    conv2 = Add()([net, conv2])
    conv2 = BatchNormalization(axis=channel_axis)(conv2)
    # conv2 = Activation('relu')(conv2)
    # conv2 = Concatenate(axis=channel_axis)([net, conv2])                        #1280, 20
    # conv2 = _conv_block_bn(conv2, 256, (3, 3), separble=True)                   # 512, 20

    net = _conv_block_up(conv2, 64, (3, 3), strides=(2, 2), padding='same')  # 256, 40
    conv1 = _conv_block_bn(conv1, 64, (3, 3), separble=False)  # 256,  5
    conv1 = Normalize(20, name='conv_up_4_0_normalize')(conv1)
    net = Normalize(10, name='conv_up_4_1_normalize')(net)
    conv1 = Add()([net, conv1])
    conv1 = BatchNormalization(axis=channel_axis)(conv1)
    # conv1 = Activation('relu')(conv1)
    # conv1 = Concatenate(axis=channel_axis)([net, conv1])                        # 512, 40
    # conv1 = _conv_block_bn(conv1, 256, (3, 3), separble=True)                   # 512, 40 

    net = _conv_block_up(conv1, 64, (3, 3), strides=(2, 2), padding='same')  # 64, 80
    conv0 = _conv_block_bn(conv0, 64, (3, 3), separble=False)  # 256,  5
    conv0 = Normalize(20, name='conv_up_5_0_normalize')(conv0)
    net = Normalize(10, name='conv_up_5_1_normalize')(net)
    conv0 = Add()([net, conv0])
    conv0 = BatchNormalization(axis=channel_axis)(conv0)
    # conv0 = Activation('relu')(conv0)
    # conv0 = Concatenate(axis=channel_axis)([net, conv0])                        # 128, 80
    # conv0 = _conv_block_bn(conv0, 256, (3, 3), separble=True)                   # 256, 80 

    # prediction from conv3_3
    # conv3_3_norm = Normalize(20, name='conv3_3_norm')(conv0)
    conv3_3_norm = conv0
    num_priors = 2
    conv3_3_norm_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                   name='conv3_3_norm_mbox_loc')(conv3_3_norm)
    conv3_3_norm_mbox_loc_flat = Flatten(name='conv3_3_norm_mbox_loc_flat')(conv3_3_norm_mbox_loc)
    conv3_3_norm_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                    kernel_initializer='he_normal', name='conv3_3_norm_mbox_conf')(conv3_3_norm)
    conv3_3_norm_mbox_conf_flat = Flatten(name='conv3_3_norm_mbox_conf_flat')(conv3_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 20.0, aspect_ratios=[0.5], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv3_3_norm_mbox_priorbox')
    conv3_3_norm_mbox_priorbox = priorbox(conv3_3_norm)

    # Prediction from conv4_3
    # conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv1)
    conv4_3_norm = conv1
    num_priors = 3
    conv4_3_norm_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                   name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
    conv4_3_norm_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                    kernel_initializer='he_normal', name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # Prediction from fc7
    # conv2 = Normalize(20, name='fc7_norm')(conv2)
    num_priors = 5
    fc7_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc')(conv2)
    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='fc7_mbox_conf')(conv2)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 40.0, max_size=70.0, aspect_ratios=[0.7, 0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(conv2)

    # Prediction from conv6_2
    num_priors = 6
    conv6_2_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                              name='conv6_2_mbox_loc')(conv3)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
    conv6_2_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                               kernel_initializer='he_normal', name='conv6_2_mbox_conf')(conv3)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 70.0, max_size=120.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv3)

    # Prediction from conv7_2
    num_priors = 6
    conv7_2_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                              name='conv7_2_mbox_loc')(conv4)
    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)
    conv7_2_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                               kernel_initializer='he_normal', name='conv7_2_mbox_conf')(conv4)
    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 110.0, max_size=160.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv4)

    # Prediction from conv8_2
    num_priors = 6
    conv8_2_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                              name='conv8_2_mbox_loc')(conv5)
    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)
    conv8_2_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                               kernel_initializer='he_normal', name='conv8_2_mbox_conf')(conv5)
    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 160.0, max_size=210.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv5)

    # Prediction from pool6
    num_priors = 6
    pool6_mbox_loc_flat = Dense(num_priors * 4, kernel_initializer='he_normal', name='pool6_mbox_loc_flat')(conv6)
    pool6_mbox_conf_flat = Dense(num_priors * num_classes, kernel_initializer='he_normal', name='pool6_mbox_conf_flat')(
        conv6)
    priorbox = PriorBox(img_size, 210.0, max_size=320.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape, name='pool6_reshaped')(conv6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)

    # Gather all predictions
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv3_3_norm_mbox_loc_flat,
                                                     conv4_3_norm_mbox_loc_flat,
                                                     fc7_mbox_loc_flat,
                                                     conv6_2_mbox_loc_flat,
                                                     conv7_2_mbox_loc_flat,
                                                     conv8_2_mbox_loc_flat,
                                                     pool6_mbox_loc_flat])

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv3_3_norm_mbox_conf_flat,
                                                       conv4_3_norm_mbox_conf_flat,
                                                       fc7_mbox_conf_flat,
                                                       conv6_2_mbox_conf_flat,
                                                       conv7_2_mbox_conf_flat,
                                                       conv8_2_mbox_conf_flat,
                                                       pool6_mbox_conf_flat])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv3_3_norm_mbox_priorbox,
                                                               conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               pool6_mbox_priorbox])

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc, mbox_conf, mbox_priorbox])

    model = Model(input_tensor, predictions)
    return model


def S3FD300COMPLEX(input_shape, num_classes=21):
    """SSD300 architecture - Modified Reference : Feature-Fused SSD: Fast Detection for Small Objects

    # Arguments
        input_shape: Shape of the input image,
            expected to be either (300, 300, 3) or (3, 300, 300)(not tested).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325 SSD300
        https://arxiv.org/pdf/1709.05054 Feature-Fused SSD: Fast Detection for Small Objects
    """
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1_1)
    conv1_3 = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block1_conv3')(conv1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_1')(conv1_2)
    block1 = merge([conv1_3, pool1], mode='sum', name='merge_conv_block_1')
    block1 = BatchNormalization(name='bn_block_1')(block1)
    block1 = Activation('relu', name='relu_block_1')(block1)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(block1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2_1)
    conv2_3 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block2_conv3')(conv2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_1')(conv2_2)
    block2 = merge([conv2_3, pool2], mode='sum', name='merge_conv_block_2')
    block2 = BatchNormalization(name='bn_block_2')(block2)
    block2 = Activation('relu', name='relu_block_2')(block2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(block2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3_2)
    conv3_4 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block3_conv4')(conv3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_1')(conv3_3)
    block3 = merge([conv3_4, pool3], mode='sum', name='merge_conv_block_3')
    block3 = BatchNormalization(name='bn_block_3')(block3)
    block3 = Activation('relu', name='relu_block_3')(block3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(block3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4_2)
    conv4_4 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block4_conv4')(conv4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_1')(conv4_3)
    block4 = merge([conv4_4, pool4], mode='sum', name='merge_conv_block_4')
    block4 = BatchNormalization(name='bn_block_4')(block4)
    block4 = Activation('relu', name='relu_block_4')(block4)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(block4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5_2)
    conv5_4 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='block5_conv4')(conv5_3)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='block5_pool_1')(conv5_3)
    block5 = merge([conv5_4, pool5], mode='sum', name='merge_conv_block_5')
    block5 = BatchNormalization(name='bn_block_5')(block5)
    block5 = Activation('relu', name='relu_block_5')(block5)

    # FC6
    fc6 = Conv2D(1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(block5)
    # FC7
    fc7 = Conv2D(1024, kernel_size=(1, 1), activation='relu', padding='same', name='fc7')(fc6)
    # Block 6
    conv6_1 = Conv2D(256, kernel_size=(1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(
        conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_padding = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(
        conv7_padding)
    # Block 8
    conv8_1 = Conv2D(128, kernel_size=(1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(
        conv8_1)

    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # prediction from conv3_3
    conv3_3_norm = Normalize(20, name='conv3_3_norm')(conv3_3)
    num_priors = 2
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv3_3_norm_mbox_loc')(conv3_3_norm)
    conv3_3_norm_mbox_loc = x
    flatten = Flatten(name='conv3_3_norm_mbox_loc_flat')
    conv3_3_norm_mbox_loc_flat = flatten(conv3_3_norm_mbox_loc)
    name = 'conv3_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv3_3_norm)
    conv3_3_norm_mbox_conf = x
    flatten = Flatten(name='conv3_3_norm_mbox_conf_flat')
    conv3_3_norm_mbox_conf_flat = flatten(conv3_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 20.0, aspect_ratios=[0.5], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv3_3_norm_mbox_priorbox')
    conv3_3_norm_mbox_priorbox = priorbox(conv3_3_norm)

    # Prediction from conv4_3
    conv4_3_norm = Normalize(20, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc = x
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    conv4_3_norm_mbox_loc_flat = flatten(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf = x
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    conv4_3_norm_mbox_conf_flat = flatten(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, aspect_ratios=[0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # Prediction from fc7
    num_priors = 5
    fc7_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    flatten = Flatten(name='fc7_mbox_loc_flat')
    fc7_mbox_loc_flat = flatten(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(fc7)
    flatten = Flatten(name='fc7_mbox_conf_flat')
    fc7_mbox_conf_flat = flatten(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 40.0, max_size=70.0, aspect_ratios=[0.7, 0.5, 0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(fc7)

    # Prediction from conv6_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc = x
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    conv6_2_mbox_loc_flat = flatten(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv6_2)
    conv6_2_mbox_conf = x
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    conv6_2_mbox_conf_flat = flatten(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 70.0, max_size=120.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv6_2)

    # Prediction from conv7_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc = x
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    conv7_2_mbox_loc_flat = flatten(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv7_2)
    conv7_2_mbox_conf = x
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    conv7_2_mbox_conf_flat = flatten(conv7_2_mbox_conf)
    priorbox = PriorBox(img_size, 110.0, max_size=160.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv7_2_mbox_priorbox')
    conv7_2_mbox_priorbox = priorbox(conv7_2)

    # Prediction from conv8_2
    num_priors = 6
    x = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc = x
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    conv8_2_mbox_loc_flat = flatten(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name=name)(conv8_2)
    conv8_2_mbox_conf = x
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    conv8_2_mbox_conf_flat = flatten(conv8_2_mbox_conf)
    priorbox = PriorBox(img_size, 160.0, max_size=210.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv8_2_mbox_priorbox')
    conv8_2_mbox_priorbox = priorbox(conv8_2)

    # Prediction from pool6
    num_priors = 6
    x = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(pool6)
    pool6_mbox_loc_flat = x
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x = Dense(num_priors * num_classes, name=name)(pool6)
    pool6_mbox_conf_flat = x
    priorbox = PriorBox(img_size, 210.0, max_size=320.0, aspect_ratios=[0.7, 0.5, 0.4, 0.33],
                        variances=[0.1, 0.1, 0.2, 0.2],
                        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape,
                             name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = priorbox(pool6_reshaped)

    # Gather all predictions
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv3_3_norm_mbox_loc_flat,
                                                     conv4_3_norm_mbox_loc_flat,
                                                     fc7_mbox_loc_flat,
                                                     conv6_2_mbox_loc_flat,
                                                     conv7_2_mbox_loc_flat,
                                                     conv8_2_mbox_loc_flat,
                                                     pool6_mbox_loc_flat])

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv3_3_norm_mbox_conf_flat,
                                                       conv4_3_norm_mbox_conf_flat,
                                                       fc7_mbox_conf_flat,
                                                       conv6_2_mbox_conf_flat,
                                                       conv7_2_mbox_conf_flat,
                                                       conv8_2_mbox_conf_flat,
                                                       pool6_mbox_conf_flat])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv3_3_norm_mbox_priorbox,
                                                               conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               pool6_mbox_priorbox])

    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4),
                       name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes),
                        name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax',
                           name='mbox_conf_final')(mbox_conf)
    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                                                           mbox_conf,
                                                           mbox_priorbox])

    model = Model(input_tensor, predictions)
    return model


if __name__ == '__main__':
    classes = ['vietnam', 'alphabet', 'number', 'symbol']

    NUM_CLASSES = len(classes) + 1
    width_size = 320
    height_size = 320
    channels = 3
    input_shape = (width_size, height_size, channels)

    model = SSD_AICR_v01(input_shape, NUM_CLASSES)
    model.summary()
    from keras.utils import plot_model

    plot_model(model, show_shapes=True, to_file='SSD_AICR_v01.png')
