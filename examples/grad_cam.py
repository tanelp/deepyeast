import numpy as np
import keras.backend as K
from keras.layers import Activation, GlobalAveragePooling2D
from skimage import exposure
import matplotlib.pyplot as plt

from deepyeast.dataset import load_data
from deepyeast.utils import preprocess_input
from deepyeast.models import DeepYeast

### select model
model = DeepYeast()
model.load_weights("weights/deepyeast-weights-22-0.902.hdf5")

### plot image
def stretch_contrast(img):
    p2, p98 = np.percentile(img, (2, 98))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale

def add_third_dimension(x):
    b_channel = np.zeros((64, 64, 1), dtype=np.uint8)
    x = np.append(x, b_channel, axis=-1)
    return x

x_val, y_val = load_data("val")

i = 0
x = x_val[i]
x = add_third_dimension(x)
x = stretch_contrast(x)

plt.imshow(x)

### plot activation map
def grad_cam(x, class_ix, layer):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    last_conv_features = model.get_layer(layer).output
    class_prob = model.get_layer("predictions").output

    grads = K.gradients(class_prob[:, class_ix], last_conv_features)[0]
    weights = GlobalAveragePooling2D()(Activation("relu")(grads))

    grad_cam_fn = K.function([model.input, K.learning_phase()], [last_conv_features, weights])
    learning_phase = 0
    out = grad_cam_fn([x, learning_phase])

    heatmap = out[0] * out[1].reshape(1, 1, 1, out[1].shape[-1])
    heatmap = np.sum(heatmap, axis=3)

    return heatmap

i = 0
x = x_val[i]
class_ix = 0
cam = grad_cam(x, class_ix, "relu3_4")
plt.matshow(cam[0])
