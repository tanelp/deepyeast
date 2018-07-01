import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from deepyeast.dataset import load_data
from deepyeast.utils import preprocess_input
from deepyeast.models import DeepYeast

class LearningRateFinder:
    def __init__(self, model):
        self.model = model
        self.lrs = []
        self.losses = []

    def find(self, x_train, y_train, batch_size=64, iters=100, start_lr=1e-5, end_lr=10.0):
        self.lr_mult = (end_lr / start_lr) ** (1.0 / iters)
        K.set_value(self.model.optimizer.lr, start_lr)

        iters_per_epoch = x_train.shape[0] // batch_size
        for i in xrange(iters):
            # get batch
            j = i % (iters_per_epoch - 1)
            ix_start = j * batch_size
            ix_end = (j + 1) * batch_size
            x = x_train[ix_start:ix_end]
            y = y_train[ix_start:ix_end]

            # do 1 step of training
            loss = self.model.train_on_batch(x, y)

            # log metrics
            self.losses.append(loss)
            lr = K.get_value(self.model.optimizer.lr)
            self.lrs.append(lr)

            # stop training if loss too large
            if np.isnan(loss) or np.isinf(loss) or 5*np.min(loss) < loss:
                print("Invalid loss, terminating training")
                break

            # increase lr
            lr *= self.lr_mult
            K.set_value(self.model.optimizer.lr, lr)

    def plot(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale("log")
        plt.xlabel("learning rate (log scale)")
        plt.ylabel("loss")

if __name__ == "__main__":
    # set up data
    x_train, y_train = load_data("train")

    num_classes = 12
    y_train = keras.utils.to_categorical(y_train, num_classes)

    x_train = preprocess_input(x_train)

    # set up model
    model = DeepYeast()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD())

    # search for lr
    lr_finder = LearningRateFinder(model)
    lr_finder.find()
    lr_finder.plot()
