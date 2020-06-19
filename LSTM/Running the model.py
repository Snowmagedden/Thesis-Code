#!/usr/bin/env python
# coding: utf-8


import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout, Flatten, LSTM
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, LeakyReLU
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


with open("lst_120.txt", "rb") as fp:   # Unpickling
    lst = pickle.load(fp)
with open("label_lst_120.txt", "rb") as fp:   # Unpickling
    label_lst = pickle.load(fp)


#transforming array into numpy array
lst_array = np.array(lst)
label_lst_array = np.array(label_lst)
label_lst_array = label_lst_array.astype(int)
label_lst_array-= 1   # otherwise it does not recognise the label 5 because it starts from 0


#splitting the train and test sets
X_train =lst_array[0:274]
X_test = lst_array[274:370]
y_train = label_lst_array[0:274]
y_test = label_lst_array[274:370]


#building the model
def lstm():
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(32, return_sequences=False,
                       input_shape=(120,2048)))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(5, activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.002),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        return model


model = lstm()
tf.keras.utils.plot_model(model, to_file='model.png', rankdir = 'LR')


history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),batch_size=32)


#Accuracy vs test accuracy graph
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


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs test loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()