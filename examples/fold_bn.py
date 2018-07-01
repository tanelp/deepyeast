import numpy as np
from keras.models import Model, load_model
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout
import keras.backend as K

def DeepYeast_wobn():
    input_shape = (64, 64, 2)
    img_input = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='conv1_1')(img_input)
    #x = BatchNormalization(name='bn1_1')(x)
    x = Activation('relu', name='relu1_1')(x)
    x = Conv2D(64, (3, 3), padding='same', name='conv1_2')(x)
    #x = BatchNormalization(name='bn1_2')(x)
    x = Activation('relu', name='relu1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='conv2_1')(x)
    #x = BatchNormalization(name='bn2_1')(x)
    x = Activation('relu', name='relu2_1')(x)
    x = Conv2D(128, (3, 3), padding='same', name='conv2_2')(x)
    #x = BatchNormalization(name='bn2_2')(x)
    x = Activation('relu', name='relu2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='conv3_1')(x)
    #x = BatchNormalization(name='bn3_1')(x)
    x = Activation('relu', name='relu3_1')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_2')(x)
    #x = BatchNormalization(name='bn3_2')(x)
    x = Activation('relu', name='relu3_2')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_3')(x)
    #x = BatchNormalization(name='bn3_3')(x)
    x = Activation('relu', name='relu3_3')(x)
    x = Conv2D(256, (3, 3), padding='same', name='conv3_4')(x)
    #x = BatchNormalization(name='bn3_4')(x)
    x = Activation('relu', name='relu3_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(512, name='ip1')(x)
    #x = BatchNormalization(name='bn4')(x)
    x = Activation('relu', name='relu4')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, name='ip2')(x)
    #x = BatchNormalization(name='bn5')(x)
    x = Activation('relu', name='relu5')(x)
    x = Dropout(0.5)(x)
    x = Dense(12, name='ip3')(x)
    x = Activation('softmax', name='predictions')(x)
    # Create model
    model = Model(img_input, x, name='deepyeast-original')
    return model

def fold_batch_norm(conv_layer, bn_layer):
    """Fold the batch normalization parameters into the weights for
       the previous layer."""
    conv_weights = conv_layer.get_weights()[0]
    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]
    epsilon = 1e-3
    new_weights = conv_weights * gamma / np.sqrt(variance + epsilon)
    new_bias = beta - mean * gamma / np.sqrt(variance + epsilon)
    return new_weights, new_bias

bn_model = load_model("deepyeast-weights-22-0.902.hdf5")
wobn_model = DeepYeast_wobn()

wobn_model.get_layer("conv1_1").set_weights(fold_batch_norm(bn_model.get_layer("conv1_1"), bn_model.get_layer("bn1_1")))
wobn_model.get_layer("conv1_2").set_weights(fold_batch_norm(bn_model.get_layer("conv1_2"), bn_model.get_layer("bn1_2")))
wobn_model.get_layer("conv2_1").set_weights(fold_batch_norm(bn_model.get_layer("conv2_1"), bn_model.get_layer("bn2_1")))
wobn_model.get_layer("conv2_2").set_weights(fold_batch_norm(bn_model.get_layer("conv2_2"), bn_model.get_layer("bn2_2")))
wobn_model.get_layer("conv3_1").set_weights(fold_batch_norm(bn_model.get_layer("conv3_1"), bn_model.get_layer("bn3_1")))
wobn_model.get_layer("conv3_2").set_weights(fold_batch_norm(bn_model.get_layer("conv3_2"), bn_model.get_layer("bn3_2")))
wobn_model.get_layer("conv3_3").set_weights(fold_batch_norm(bn_model.get_layer("conv3_3"), bn_model.get_layer("bn3_3")))
wobn_model.get_layer("conv3_4").set_weights(fold_batch_norm(bn_model.get_layer("conv3_4"), bn_model.get_layer("bn3_4")))
wobn_model.get_layer("ip1").set_weights(fold_batch_norm(bn_model.get_layer("ip1"), bn_model.get_layer("bn4")))
wobn_model.get_layer("ip2").set_weights(fold_batch_norm(bn_model.get_layer("ip2"), bn_model.get_layer("bn5")))

wobn_model.get_layer("ip3").set_weights(bn_model.get_layer("ip3").get_weights())

image_data = np.random.random((1, 64, 64, 2)).astype('float32')
preds1 = bn_model.predict(image_data)
preds2 = wobn_model.predict(image_data)

#
get_last_conv_layer_output_bn = K.function([bn_model.layers[0].input, K.learning_phase()],
                                   [bn_model.get_layer("bn1_1").output])
get_last_conv_layer_output_wobn = K.function([wobn_model.layers[0].input, K.learning_phase()],
                                   [wobn_model.get_layer("conv1_1").output])

# output in test mode = 0
out1 = get_last_conv_layer_output_bn([image_data, 0])[0]
out2 = get_last_conv_layer_output_wobn([image_data, 0])[0]

out1 = get_last_conv_layer_output_bn([image_data, 1])[0]
out2 = get_last_conv_layer_output_wobn([image_data, 1])[0]

# output in train mode = 1
layer_output = get_3rd_layer_output([x, 1])[0]

wobn_model.save("deepyeast-weights-22-0.902-bnfolded.hdf5")
