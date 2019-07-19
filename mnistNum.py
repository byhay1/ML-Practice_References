#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 22:35:57 2019

@author: beehive
"""
# MNIST using Tensorflow/Keras
#
#---------------------
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#-------------VARIABLES-------------#
# 28x28 images of handwritten digits from 0 to 9
mnist = tf.keras.datasets.mnist 

# load training data from mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# scale (normalize) the data - changes the data from the current classifying set (0-255)
#to a normalized value (0<= k <=1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#-------------PLOT-------------#
# show the first number in the training dataset
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()

#-------------MODEL-------------#
# fwd propagation through sequence
model = tf.keras.models.Sequential()

# change the 28x28 image array into a 28x1 using flatten
#used for the input layer
model.add(tf.keras.layers.Flatten())

# hidden layer one, passing through parameters with (128 units(neurons), activation function=rectified linear)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# hidden layer two  passing through parameters with (128 units(neurons), activation function=rectified linear)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer passing through parameters with (10 units(neurons), activation function=softmax)
#10 is chosen because we have 10 numbers; while softmax is chosen because this is a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#-------------TRAIN-------------#
# Define the parameters for the training of the model
#optimizer is used to minimize loss, loss... , metrics is how the model is perceived. 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train model 
model.fit(x_train, y_train, epochs=3) 

#-------------LOSS/ACCURACY-------------#
# Check for over and underfitting by evaluating the loss and accuracy of the training
#checked against your test set. Note: if if the delta is large then the model is most likely overfitting
value_loss, value_acc = model.evaluate(x_test, y_test) 

#-------------INITIATE-------------#
if __name__ == '__main__': 
    #pass
    #print(tf.__version__)
    #print(x_train[0])
    #print(x_train,x_test)
    print(value_loss, value_acc)
    

############EXTRA##############
###############################
#Save Model, Set Model, Predict
###############################
#model.save('tf_keras_mnist_numNet')
#model_new = tf.keras.models.load_model('tf_keras_mnist_numNet')
#predictions = model_new.predict([x_test]) # Spits out one-hot arrays (probability distribution)
#print(np.argmax(predictions[0])) # shows what the model predicts at first index in array.
#plt.imshow(x_test[0]) # check to see if model predicted correctly
#plt.show()
###############################