import keras
from keras.preprocessing.image import ImageDataGenerator

from deepyeast.dataset import load_data
from deepyeast.utils import preprocess_input, mixup_generator
from deepyeast.models import DeepYeast

# set up data
x_train, y_train = load_data("train")
x_val, y_val = load_data("val")

num_classes = 12
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# set up model
model = DeepYeast()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

filepath="weights-{epoch:02d}-{val_acc:.3f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr]

dataaug = ImageDataGenerator(width_shift_range=10. / 64,
                             height_shift_range=10. / 64,
                             horizontal_flip=True)
# training loop
batch_size = 64
epochs = 300
steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size)))
model.fit_generator(mixup_generator(x_train, y_train, batch_size, alpha=0.2, dataaug=dataaug),
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)
