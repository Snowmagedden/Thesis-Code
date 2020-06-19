#!/usr/bin/env python
# coding: utf-8


import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
import pandas as pd


final = pd.read_csv('final.csv')


# creating an empty list of image arrays of the image frames
image_array = []

# for loop to read and store frames
for i in tqdm(range(final.shape[0])):
    # loading the image and keeping the target size as (224,224,3)
    img = image.load_img('image frames/'+final['path'][i], target_size=(224,224,3))    # also try 250 x 250
    # converting it to array
    img = image.img_to_array(img)
    # normalizing the pixel value
    img = img/255                                    
    # appending the image to the train_image list
    image_array.append(img)


# converting the list to numpy array and splitting into X_train and X_test
X_train = np.array(image_array[0:6576])
X_test = np.array(image_array[6576:8880])



y = final['engagement rating']
y = y.astype(int)
y = np.array(y)
y-= 1                     #We need -1 because python starts at 0


#splitting the y_train and y_test
y_train = y[0:6576]
y_test = y[6576:8880]


#Assigning ResNet50 to a variable 
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


#Extracting features from the images using ResNet
#This is very computationally demanding and will take a long time
X_train = base_model.predict(X_train)
X_test = base_model.predict(X_test)


# normalizing the pixel values
max = X_train.max()
X_train = X_train/max
X_test = X_test/max


#Building the model
def build_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None,
                             input_shape=[7,7,2048]))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=1, strides=2))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    return model


model = build_model()
mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
model.compile(loss='sparse_categorical_crossentropy',optimizer=tf.keras.optimizers.Adadelta(lr=0.002),metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=128)


history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss_values = history_dict['loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, acc, 'r', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Test acc') 
plt.title('Training and test accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy') 
plt.legend()
plt.show()