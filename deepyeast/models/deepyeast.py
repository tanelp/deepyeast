from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout

def preprocess_input(x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

def DeepYeast():

    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(img_input)
    x = BatchNormalization(name='bn1_1')(x)
    x = Activation('relu', name='relu1_1')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
    x = BatchNormalization(name='bn1_2')(x)
    x = Activation('relu', name='relu1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
    x = BatchNormalization(name='bn2_1')(x)
    x = Activation('relu', name='relu2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
    x = BatchNormalization(name='bn2_2')(x)
    x = Activation('relu', name='relu2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
    x = BatchNormalization(name='bn3_1')(x)
    x = Activation('relu', name='relu3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
    x = BatchNormalization(name='bn3_2')(x)
    x = Activation('relu', name='relu3_2')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    x = BatchNormalization(name='bn3_3')(x)
    x = Activation('relu', name='relu3_3')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_4')(x)
    x = BatchNormalization(name='bn3_4')(x)
    x = Activation('relu', name='relu3_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(512, name='ip1')(x)
    x = BatchNormalization(name='bn4')(x)
    x = Activation('relu', name='relu4')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, name='ip2')(x)
    x = BatchNormalization(name='bn5')(x)
    x = Activation('relu', name='relu5')(x)
    x = Dropout(0.5)(x)
    x = Dense(12, name='ip3')(x)
    x = Activation('softmax', name='predictions')(x)

    # Create model
    model = Model(img_input, x, name='deepyeast-original')

    return model
