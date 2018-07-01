from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization, Activation, Dense
from keras.applications.mobilenet import DepthwiseConv2D

def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

def _depthwise_separable_conv_block(x, pointwise_conv_channels, alpha=1.0, depth_multiplier=1, strides=1, block_id=1):
    name = 'conv_block_{}'.format(block_id)
    pointwise_conv_channels = int(alpha*pointwise_conv_channels)

    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name=name+'_dw_conv')(x)
    x = BatchNormalization(name=name+'_dw_bn')(x)
    x = Activation('relu', name=name+'_dw_relu')(x)
    x = Conv2D(pointwise_conv_channels, (1, 1), padding='same', strides=1, use_bias=False, name=name+'_pw_conv')(x)
    x = BatchNormalization(name=name+'_pw_bn')(x)
    x = Activation('relu', name=name+'_pw_relu')(x)
    return x

def MobileNet():
    alpha = 1.0
    depth_multiplier = 1
    num_initial_channels = 32

    num_classes = 12
    
    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape)

    x = Conv2D(int(num_initial_channels*alpha), (3, 3), strides=2, padding='same', use_bias=False, name='initial_conv')(img_input)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)

    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), (1024, 2), (1024, 1)]
    for i, (channels, strides) in enumerate(cfg):
        x = _depthwise_separable_conv_block(x, channels, alpha=alpha, depth_multiplier=depth_multiplier, strides=strides, block_id=i)

    # classification head
    x = GlobalAveragePooling2D(name='final_global_pool')(x)
    x = Dense(num_classes, activation='softmax', name='prob')(x)

    model = Model(img_input, x, name='mobilenet')

    return model
