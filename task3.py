#Image Classification
#importing libraries
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

#loading the datasets
x_train= np.loadtxt('datasets/input.csv', delimiter=',')
y_train= np.loadtxt('datasets/labels.csv', delimiter=',')
x_test= np.loadtxt('datasets/input_test.csv', delimiter=',')
y_test= np.loadtxt('datasets/labels_test.csv', delimiter=',')

#reshaping the size 
x_train= x_train.reshape(len(x_train), 100, 100, 3)
y_train= y_train.reshape(len(y_train), 1)
x_test= x_test.reshape(len(x_test), 100, 100, 3)
y_test= y_test.reshape(len(y_test), 1)

#data overview
print("Shape of xtrain: ", x_train.shape)
print("Shape of ytrain: ", y_train.shape)
print("Shape of xtest: ", x_test.shape)
print("Shape of ytest: ", y_test.shape)

#taking a random img from x_train dataset and plotting it
idx= random.randint(0, len(x_train))
plt.imshow(x_train[idx, :])
plt.show()

#creating the model
model= Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),
    
    Conv2D(32,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size= 64)
model.evaluate(x_test, y_test)

#taking a random img from x_test and plotting it
idx2= random.randint(0, len(y_test))
plt.imshow(x_test[idx2, :])
plt.show()

#make predictions
y_pred= model.predict(x_test[idx2, :].reshape(1,100,100,3))
y_pred= y_pred > 0.5
if y_pred==0:
    pred= 'dog'
else:
    pred='cat'
print("Our model says it is a :", pred)
