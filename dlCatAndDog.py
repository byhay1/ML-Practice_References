#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:44:34 2019

@author: beehive
"""

# Keras/TensorFlow Convolutional Neural Network
# Kaggle Cats and Dogs
#
#---------------------
import numpy as np
import matplotlib.pyplot as plt
import os 
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


#-------------FUNCTIONS-------------#



#-------------VARIABLES-------------#
# define your data path and categories.  
DATADIR = "/home/beehive/Documents/ML-Cats-n-Dogs/PetImages"
CATEGORIES = ["Dog", "Cat"]

# test - define your normalized image size
IMG_SIZE = 100
#new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


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

# read out X
pickle_in = open("X.pickle", "rb")

##-------------PLOT-------------#
# print the new image size
#plt.imshow(new_array, cmap = 'gray')
#plt.show()

#-------------MODEL-------------#

#-------------TRAIN-------------#
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

create_training_data()
#-------------LOSS/ACCURACY-------------#

#-------------NON_FUNCT-------------#
# test -  iterate through all data in path for cats and dogs. 
#for category in CATEGORIES: 
#    path = os.path.join(DATADIR, category) # path to cats or dogs dir
#    for img in os.listdir(path): 
#        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
#        plt.imshow(img_array, cmap="gray")
#        plt.show
#        break 
#    break

# shuffle the training data so that the neural network does not accidently force predict
#based on a predictable reoccuring data set. 
random.shuffle(training_data)

# check labels for training data are correct
#for sample in training_data[:10]: 
#   print(sample[1])

# take the shuffled data and pack into variables before NN analysis
for features, label in training_data: 
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # '-1' is for all features. '1' is for grayscale
######################################

######################################
#-------------INITIATE-------------#
if __name__ == '__main__': 
    #pass
    #print(img_array) # test - see image value
    #print(img_array.shape) # test - see image.shape
    print(len(training_data))
    print(X[1])
    