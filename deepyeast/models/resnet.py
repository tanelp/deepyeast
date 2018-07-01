from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, add
from keras import regularizers

def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

def bn_relu_conv(x, num_channels, kernel_size=3, strides=1, weight_decay=1e-4, block_name=None):
    x = BatchNormalization(name=block_name+'_bn')(x)
    x = Activation('relu', name=block_name+'_relu')(x)
    x = Conv2D(num_channels, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(weight_decay), name=block_name+'_conv')(x)
    return x

def ResNet50():
    weight_decay = 1e-4
    num_channels_in = 16
    channel_multiplier = 2
    num_blocks = 4
    num_sub_blocks = 4

    num_classes = 12

    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape)

    x = Conv2D(num_channels_in, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name='initial_conv')(img_input)

    for i in range(num_blocks):
        num_channels_out = channel_multiplier * num_channels_in
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            name = 'block_{}_layer_{}'.format(i, j)
            y = bn_relu_conv(x, num_channels_in, kernel_size=1, strides=strides, weight_decay=weight_decay, block_name=name+'_down')
            y = bn_relu_conv(y, num_channels_in, kernel_size=3, strides=1, weight_decay=weight_decay, block_name=name)
            y = bn_relu_conv(y, num_channels_out, kernel_size=1, strides=1, weight_decay=weight_decay, block_name=name+'_up')
            if j == 0:
                x = Conv2D(num_channels_out, kernel_size=1, strides=strides, kernel_regularizer=regularizers.l2(weight_decay))(x)
            x = add([x, y], name='block_{}_{}_add'.format(i, j))
        num_channels_in = num_channels_out

    # classification head
    x = BatchNormalization(name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)
    x = GlobalAveragePooling2D(name='final_global_pool')(x)
    #x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay), name='prob')(x)
    x = Dense(num_classes, activation='softmax', name='prob')(x)

    model = Model(img_input, x, name='resnet50')

    return model
