import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import GlobalAveragePooling2D, AveragePooling2D
from keras.layers import Concatenate, Multiply, Add, Permute
from keras.layers import Input, Reshape, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.convolutional import MaxPooling2D
from keras import initializers, regularizers

from model.ssd_layers import Normalize, PriorBox
from model.coord_layer import CoordinateChannel2D


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
    net = Dropout(0.2)(net)
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

    net = _conv_block_bn(input, 64, (3, 3))  # 32,  160
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


def SSD_AICR(input_shape, num_classes=21):  # inception v4 style with Seperable Convolutions
    channel_axis = -1

    # Block 1
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])

    ## Stem layers + feature extraction conv0 for aicr
    net, conv0 = _block_stem(input_tensor)  # (256,  40), (64, 80)
    conv0 = squeeze_excite_block(conv0)

    tem_i = 0
    while tem_i < 2:
        net = _block_inception_a(net)  # 384,  40
        net = squeeze_excite_block(net)
        tem_i = tem_i + 1
    ## feature extraction - 1 : conv1
    conv1 = net  # 384,  40

    ## reduction layer
    ## feature extraction - 2 : conv2
    net = _block_reduction_a(net)  # (384, 40)→(1024, 20)

    tem_i = 0
    while tem_i < 3:
        net = _block_inception_b(net)  # 1024,  20
        net = squeeze_excite_block(net)
        tem_i = tem_i + 1

    conv2 = net  # 1024,  20

    ## reduction layer
    ## feature extraction - 3 : conv3
    net = _block_reduction_b(net)  # (1024, 20)→(1536, 10)
    conv3 = net  # 1536,  10

    # ## feature extraction - 4 : conv4
    tem_i = 0
    while tem_i < 2:
        net = _block_inception_c(net)  # 1536,  10 # tuan anh: conv3->net
        net = squeeze_excite_block(net)
        tem_i = tem_i + 1

    conv4 = _conv_block_bn(net, 256, (3, 3), strides=(2, 2), separble=True)  # 512,  5

    # prediction from conv3_3 : 80x80X 256 - Conv0 + Up(Conv1)
    # Prediction from conv4_3 : 40x40x 512 - Conv1 + Up(Conv2)
    # Prediction from fc7     : 20x20x1024 - Conv2 + Up(Conv3)
    # Prediction from conv6_2 : 10x10x 512 - Conv3 + Up(Conv4)

    # upsampling Block
    net = _conv_block_up(conv4, 96, (3, 3), strides=(2, 2), padding='same')  # 256, 10
    conv3 = _conv_block_bn(conv3, 96, (3, 3), separble=False)  # 256,  5
    conv3 = Normalize(20, name='conv_up_2_0_normalize')(conv3)
    net = Normalize(10, name='conv_up_2_1_normalize')(net)
    conv3 = Add()([net, conv3])
    conv3 = BatchNormalization(axis=channel_axis)(conv3)

    net = _conv_block_up(conv3, 96, (3, 3), strides=(2, 2), padding='same')  # 256, 20
    conv2 = _conv_block_bn(conv2, 96, (3, 3), separble=False)  # 256,  5
    conv2 = Normalize(20, name='conv_up_3_0_normalize')(conv2)
    net = Normalize(10, name='conv_up_3_1_normalize')(net)
    conv2 = Add()([net, conv2])
    conv2 = BatchNormalization(axis=channel_axis)(conv2)

    net = _conv_block_up(conv2, 96, (3, 3), strides=(2, 2), padding='same')  # 256, 40
    conv1 = _conv_block_bn(conv1, 96, (3, 3), separble=False)  # 256,  5
    conv1 = Normalize(20, name='conv_up_4_0_normalize')(conv1)
    net = Normalize(10, name='conv_up_4_1_normalize')(net)
    conv1 = Add()([net, conv1])
    conv1 = BatchNormalization(axis=channel_axis)(conv1)

    net = _conv_block_up(conv1, 96, (3, 3), strides=(2, 2), padding='same')  # 64, 80
    conv0 = _conv_block_bn(conv0, 96, (3, 3), separble=False)  # 256,  5
    conv0 = Normalize(20, name='conv_up_5_0_normalize')(conv0)
    net = Normalize(10, name='conv_up_5_1_normalize')(net)
    conv0 = Add()([net, conv0])
    conv0 = BatchNormalization(axis=channel_axis)(conv0)

    # prediction from conv3_3
    conv3_3_norm = conv0
    num_priors = 3
    conv3_3_norm_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                   name='conv3_3_norm_mbox_loc')(conv3_3_norm)
    conv3_3_norm_mbox_loc_flat = Flatten(name='conv3_3_norm_mbox_loc_flat')(conv3_3_norm_mbox_loc)
    conv3_3_norm_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                    kernel_initializer='he_normal', name='conv3_3_norm_mbox_conf')(conv3_3_norm)
    conv3_3_norm_mbox_conf_flat = Flatten(name='conv3_3_norm_mbox_conf_flat')(conv3_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 8.0, max_size=16.0, aspect_ratios=[0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv3_3_norm_mbox_priorbox')
    conv3_3_norm_mbox_priorbox = priorbox(conv3_3_norm)

    # Prediction from conv4_3
    conv4_3_norm = conv1
    num_priors = 3
    conv4_3_norm_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                                   name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
    conv4_3_norm_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                                    kernel_initializer='he_normal', name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    priorbox = PriorBox(img_size, 16.0, max_size=30.0, aspect_ratios=[0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='conv4_3_norm_mbox_priorbox')
    conv4_3_norm_mbox_priorbox = priorbox(conv4_3_norm)

    # Prediction from fc7
    num_priors = 3
    fc7_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc')(conv2)
    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
    fc7_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='fc7_mbox_conf')(conv2)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)
    priorbox = PriorBox(img_size, 30.0, max_size=60.0, aspect_ratios=[0.33], variances=[0.1, 0.1, 0.2, 0.2],
                        name='fc7_mbox_priorbox')
    fc7_mbox_priorbox = priorbox(conv2)

    # Prediction from conv6_2
    num_priors = 3
    conv6_2_mbox_loc = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                              name='conv6_2_mbox_loc')(conv3)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
    conv6_2_mbox_conf = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same',
                               kernel_initializer='he_normal', name='conv6_2_mbox_conf')(conv3)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
    priorbox = PriorBox(img_size, 60.0, max_size=130.0, aspect_ratios=[0.33],
                        variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    conv6_2_mbox_priorbox = priorbox(conv3)

    # Gather all predictions
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv3_3_norm_mbox_loc_flat,
                                                     conv4_3_norm_mbox_loc_flat,
                                                     fc7_mbox_loc_flat,
                                                     conv6_2_mbox_loc_flat])

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv3_3_norm_mbox_conf_flat,
                                                       conv4_3_norm_mbox_conf_flat,
                                                       fc7_mbox_conf_flat,
                                                       conv6_2_mbox_conf_flat])

    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv3_3_norm_mbox_priorbox,
                                                               conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox])

    num_boxes = 0
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1] // 4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1] // 4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc, mbox_conf, mbox_priorbox])

    return Model(input_tensor, predictions)


if __name__ == '__main__':
    classes = ['vietnam', 'alphabet', 'number', 'symbol']

    NUM_CLASSES = len(classes) + 1
    width_size = 320
    height_size = 320
    channels = 3
    input_shape = (width_size, height_size, channels)

    model = SSD_AICR(input_shape, NUM_CLASSES)
    model.summary()
    from keras.utils import plot_model

    plot_model(model, show_shapes=True, to_file='SSD_AICR_vn_new.pdf')
