#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:15:40 2019

@author: beehive
@original : johnwittenauer.net & curious insight 
"""

# Determine chance of admission from two exams using Logistic Regression
#---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pathlib import Path


#-------------FUNCTIONS-------------#
# The "activation" function. converts a continuous input into a value between zero and one (or class probability)
#the liklihod that the input example should be classified positively for label prediction. 
#formula is f(x) = 1 / (1 + e^(-x)) where x is your weight.T * x(feature vector)... dot product.
def sigmoid(zed):
    return 1. / (1 + np.exp(-zed))

# cost error function through logistic regression
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))

# define gradient to reduce the training error (or cost)
#computes one step of the gradient
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)    
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters): 
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)
        
    return grad

# prediction for a dataset using the learned parameters theta. 
#used to score the training accuracy of the classifier
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

#-------------VARIABLES-------------#
path = Path('/home/beehive/Documents/ML-SelfStudy/Linear-Regression/ex2data1.txt')
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
see_data = data.head()

# set the variables of the two classes
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

# set a range from -10 to 10 with a step of one to see sigmoid function
nums = np.arange(-10,10,step=1)

# add a ones column for easier multiplication
data.insert(0, 'Ones', 1)

# set X (training data) & y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# convert to numpy arays and then ititalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# instead of going through each step, use optimize from scipy to get the optimal model parameters. 
result = opt.fmin_tnc(func=cost,x0=theta, fprime=gradient, args=(X, y))
optimum = cost(result[0], X, y)

# define for use to predict the acuracy of the classifier  
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a,b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))

#-------------PLOT-------------#
#See the two scores by color coding by '0' or '1' (0 not admitted and 1 admitted)
#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
#ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
#ax.legend()
#ax.set_xlabel('Exam 1 Score')
#ax.set_ylabel('Exam 2 Score')

#See the sigmoid function based on a range from -10 to 10, wih step size of 1 (see nums). 
#fig, ax = plt.subplots(figsize=(12,8))
#ax.plot(nums, sigmoid(nums), 'r')

#-------------INITIATE-------------#
if __name__ == '__main__':
    #print(see_data)
    print(cost(theta, X, y))
    print(optimum)
    print('accuracy = {0}%'.format(accuracy))