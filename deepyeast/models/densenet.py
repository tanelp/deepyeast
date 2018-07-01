from keras.models import Model
from keras.layers import Dense, Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dropout, concatenate
import keras.backend as K
from keras import regularizers

def preprocess_input(x):
    x = x.astype('float32') / 255.
    x -= 0.5
    x *= 2.
    return x

def DenseNet40_BC():
    growth_rate = 12
    num_dense_blocks = 4
    num_layers_per_block = 9
    num_classes = 12
    weight_decay = 1e-4
    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape, name='data')

    # initial convolution
    x = Conv2D(2 * growth_rate, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name='initial_conv')(img_input)

    # dense block 1 and transition
    for i in range(num_dense_blocks - 1):
        x = _dense_block(x, num_layers_per_block, growth_rate, bottleneck=True, dropout_rate=0.2, weight_decay=weight_decay, block_id=i)
        x = _transition_block(x, compression=0.5, dropout_rate=0.2, weight_decay=1e-4, block_id=i)

    # last block doesn't have transition block
    x = _dense_block(x, num_layers_per_block, growth_rate, bottleneck=True, dropout_rate=0.2, weight_decay=weight_decay, block_id=num_dense_blocks)

    x = BatchNormalization(name='final_bn')(x)
    x = Activation('relu', name='final_relu')(x)
    x = GlobalAveragePooling2D(name='final_global_pool')(x)
    #x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay), name='prob')(x)
    x = Dense(num_classes, activation='softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet40_bc')

    return model

def _conv_block(x, num_channels, bottleneck=False, dropout_rate=None, weight_decay=1e-4, block_name=None):
    x = BatchNormalization(name=block_name+'_bn')(x)
    x = Activation('relu', name=block_name+'_relu')(x)

    if bottleneck:
        x = Conv2D(4*num_channels, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name=block_name+'_bottleneck_conv')(x)
        if dropout_rate:
            x = Dropout(dropout_rate, name=block_name+'_bottleneck_dropout')(x)
        x = BatchNormalization(name=block_name+'_bottleneck_bn')(x)
        x = Activation('relu', name=block_name+'_bottleneck_relu')(x)

    x = Conv2D(num_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name=block_name+'_conv')(x)
    if dropout_rate:
        x = Dropout(dropout_rate, name=block_name+"_dropout")(x)

    return x

def _dense_block(x, num_layers, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1e-4, block_id=None):
    for i in range(num_layers):
        conv_name = 'dense_block_{}_conv_{}'.format(block_id, i)
        concat_name = 'dense_block_{}_concat_{}'.format(block_id, i)
        cb = _conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay, conv_name)
        x = concatenate([x, cb], name=concat_name)
    return x

def _transition_block(x, compression=0.5, dropout_rate=None, weight_decay=1e-4, block_id=None):
    channel_axis = 3
    num_channels = K.int_shape(x)[channel_axis]
    num_channels = int(num_channels*compression)
    base_name = 'transition_block_{}_'.format(block_id)
    x = BatchNormalization(name=base_name+'bn')(x)
    x = Activation('relu', name=base_name+'relu')(x)
    x = Conv2D(num_channels, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name=base_name+'conv')(x)
    if dropout_rate:
        x = Dropout(dropout_rate, name=base_name+'dropout')(x)
    x = AveragePooling2D((2, 2), strides=(2, 2), name=base_name+'pool')(x)
    return x
