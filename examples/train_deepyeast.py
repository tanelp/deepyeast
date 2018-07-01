import keras

from deepyeast.dataset import load_data
from deepyeast.utils import preprocess_input
from deepyeast.models import DeepYeast

# set up data
x_val, y_val = load_data("val")
x_train, y_train = load_data("train")

num_classes = 12
y_val = keras.utils.to_categorical(y_val, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# set up model
model = DeepYeast()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

filepath="../weights-{epoch:02d}-{val_acc:.3f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr]

# training loop
batch_size = 64
epochs = 300
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)
