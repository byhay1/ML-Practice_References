#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:42:51 2019

@author: beehive
@original : johnwittenauer.net & curious insight 
"""
"""
WARNING
    Accuracy is wrongly calculated and will need revisions.
WARNING
"""
# Logistic Regression with Regularization is fine for also multiple variables (where X > 2)
#
#---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
#from pathlib import Path


#-------------FUNCTIONS-------------#
# sigmoid function 
def sigmoid(zed):
    return 1 / (1 + np.exp(-zed))

# logistic regression cost function 
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / (len(X)) + reg

# Gradient Descent, to reduce cost. 
def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)
    
    # gradient of intercept is not regularized
    grad[0,0] = np.sum(np.multiply(error, X[:, 0])) / len(X)
    
    return np.array(grad).ravel()

# define one-V-all classification where a label with q different classes results in q number 
#of classifiers(each one with binary deciding factors). 
def all_for_one(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    
    # q X (n+1) array for the params of each of the q classifiers
    all_theta = np.zeros((num_labels, params + 1))
    
    # insert a column of ones at the beginnging for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        # minimize the object function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i-1,:] = fmin.x
        
        return all_theta

# Use the trained classifier to predict a label for each imag
#compute the class probability for each class, for each training instance(vectorized), and assign the output 
#class label as the class with the highest probability.
def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrix
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each clas on each training instance 
    h = sigmoid(X * all_theta.T)

    # create array of the index with the max probability
    h_argmax = np.argmax(h, axis=1)

    # because the array was zero-indexed, we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax
    
#-------------VARIABLES-------------#
data = loadmat('/home/beehive/Documents/ML-SelfStudy/Linear-Regression/ex3data1.mat')

# define variables
rows = data['X'].shape[0]
params = data['X'].shape[1]

all_theta = np.zeros((10, params + 1))

X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0, (rows, 1))

# test taining function
t_all_theta = all_for_one(data['X'], data['y'], 10, 1)

# use predict_all to generate class predictions for each instance
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))




#-------------INITIATE-------------#
if __name__ == '__main__': 
    #pass
    print(data)
    # print(data['X'].shape, data['y'].shape)
    print(X.shape, y_0.shape, theta.shape, all_theta.shape)
    print(data['X'].shape)
    print(np.unique(data['y']))
    print(t_all_theta)
    # get accuracy
    print(" accuracy should be 97.58%, if it isn't code needs to be fixed\n",'accurary = {0}%'.format(accuracy * 100))