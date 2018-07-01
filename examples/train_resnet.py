import keras

from resnet import *
from dataset import *

x_train, y_train = load_data("train")
x_val, y_val = load_data("val")

# convert class vectors to binary class matrices
num_classes = 12
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# preprocess
x_train = preprocess_input(x_train.astype("float"))
x_val = preprocess_input(x_val.astype("float"))

model = ResNet50()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
              metrics=['accuracy'])

filepath="weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, cooldown=0, min_lr=1e-5)
callbacks_list = [checkpoint, reduce_lr]

batch_size = 64
epochs = 300
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          callbacks=callbacks_list)
