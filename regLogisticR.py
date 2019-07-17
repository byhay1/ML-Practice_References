#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:58:12 2019

@author: beehive
@original : johnwittenauer.net & curious insight 
"""
"""
WARNING
    Accuracy is wrongly calculated and will need revisions.
WARNING
"""
# Improve the logistic regression through regularization
#Regularization is a term in the cost function that causes the algorithmn to perfer simple models. 
#Regularization will be used to determine which chips should be rejected or accepted
#---------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pathlib import Path

 
#-------------FUNCTIONS-------------#
#Sigmoid Function
def sigmoid(zed):
    return 1. / (1 + np.exp(-zed))

# regularization function (for each case the regularization is added on to the previous calculation)
#used to fix overfitting. 
#Updated cost function 
def costReg (theta, X, y, learningRate): 
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1-y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    
    return np.sum(first - second) / (len(X)) + reg

# Additionally, add the regularization to the gradient function.
def gradientReg(theta, X, y, learningRate): 
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:,i])
            
    return grad
     
# prediction for a dataset using the learned parameters theta. 
#used to score the training accuracy of the classifier
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]
   
#-------------VARIABLES-------------#
path = Path('/home/beehive/Documents/ML-SelfStudy/Linear-Regression/ex2data2.txt')
data = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

degree = 5
x1 = data['Test 1']
x2 = data['Test 2']
data.insert(3, 'Ones', 1)

# set 'X' and 'y'
cols = data.shape[1]
X2 = data.iloc[:,1:cols]
y2 = data.iloc[:,0:1]

#convert to numpy arrays and initialize parameter theta
X2 = np.array(X2.values)
y2 = np.array(y2.values)
theta2 = np.zeros(3)

# set learning rate
learningRate = 1

# GD w/ Regularization call
gdC = costReg(theta2, X2, y2, learningRate)

# find optimal model parameters
result = opt.fmin_tnc(func=costReg, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))

# use for label predictions for the training data to evaluate performance
theta_min = np.matrix(result[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))


#-------------PLOT-------------#
# plot data from ex2data2 with positive as 'o' and negative as 'x'
#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
#ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='y', marker='x', label='Rejected')
#ax.legend()
#ax.set_xlabel('Test 1 score')
#ax.set_ylabel('Test 2 score')

#-------------NON_FUNCT-------------#
# data is not linear-seperable; thus, features must be constructed to be derived from polynomials
#of the features. 
#Create different polynomial features to feed into the classifier
for i in range(1, degree):
    for j in range(0, i): 
        data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
data.drop('Test 1', axis=1, inplace=True)
data.drop('Test 2', axis=1, inplace=True)

polyF = data.head()

#-------------INITIATE-------------#
if __name__ == '__main__':
    #print(polyF)
    #print(X2,'/n', y2, '/n', theta2)
    #print(np.shape(X2),'/n', np.shape(y2), '/n', np.shape(theta2))
    print(gdC)
    #print(result)
    print(" should print 91% - if it doesn't code needs to be fixed\n", 'accuracy = {}'.format(accuracy))