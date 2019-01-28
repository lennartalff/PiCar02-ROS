import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import model as m
import files_to_data as ftd

img_width, img_height = 60, 60

load_save_dir = "/home/lennartalff/catkin_ws/src/computer_calc/scripts/2500_initial_softmax.h5"

train_data_dir = "/home/lennartalff/catkin_ws/src/computer_calc/pictures_train"
validation_data_dir = "/home/lennartalff/catkin_ws/src/computer_calc/pictures_test"
nb_train_samples = 8000
nb_validation_samples = 2800
epochs = 20
batch_size = 50

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model = m.load_trained_model(load_save_dir)

# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)

# model.save_weights(r"C:\Users\Mike\Desktop\Studium\TUHH\car-firmware-master\Projekt\96softmaxkopie1.h5")

size=(60,60)
x = Image.open("/home/lennartalff/catkin_ws/src/computer_calc/pictures_train/80/img_ts_1369.jpg")
x = x.resize(size)
x = ftd.jpg_image_to_array(x, size)
x = x.reshape((1,) + x.shape)
print(x.shape)
z = model.predict(x)
predicted_class_indices=np.argmax(z,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


print(predictions)
print(validation_generator.class_indices)