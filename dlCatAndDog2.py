#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 22:24:09 2019

@author: beehive
"""

# Keras/TensorFlow Convolutional Neural Network
# Kaggle Cats and Dogs (2) 
#
#---------------------
import numpy as np
#import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


#-------------VARIABLES-------------#
# define your data path and categories.
DATADIR = "/home/beehive/Documents/ML-Cats-n-Dogs/PetImages"
CATEGORIES = ["Dog", "Cat"]

#test - define your normalized image size
IMG_SIZE = 100

X = []
y = []

# save X (feature variable) 
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# save y (labels)
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

# define feature variables and labels to be used in NN analysis
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# test - define your normalized image size
#X = X/255.0

#-------------PLOT-------------#


#-------------FUNCTIONS-------------#
# create training data by iterating through everything and putting the data into a single array
training_data = []

def create_training_data(): 
    for category in CATEGORIES: 
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category) # map to a numerical value based on the list in CATEGORIES
        for img in os.listdir(path): 
            try: 
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e: 
                pass
            
#run function - create the training data
create_training_data()

#-------------NON_FUNCT-------------#
# shuffle the training data so that the neural network does not accidently force predict
#based on a predictable reoccuring data set. 
random.shuffle(training_data)

# take the shuffled data and pack into variables before NN analysis
for features, label in training_data: 
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # '-1' is for all features. '1' is for grayscale

#-------------MODEL-------------#
# create CNN model
#layer 1
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#layer 2 
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#layer 3 (convert 3d feature maps to 1d vector)
model.add(Flatten())
model.add(Dense(64))

#output layer, final activation
model.add(Dense(1))
model.add(Activation("sigmoid"))

#-------------TRAIN-------------#


# Define the parameters for the training of the model
#optimizer is used to minimize loss, loss... , metrics is how the model is perceived.
model.compile(loss="binary_crossentropy", 
              optimizer="adam", 
              metrics=['accuracy'])

#train the model
model.fit(X, y, batch_size=32, epochs = 3, validation_split=0.1)

#-------------LOSS/ACCURACY-------------#

#-------------INITIATE-------------#
if __name__ == '__main__': 
    #pass
    print("End Training CNN Model")
