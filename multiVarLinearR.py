#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:36:03 2019

@author: beebrain
@original : johnwittenauer.net & curious insight 
"""

# Predict the selling price of a house. This is based on two variabbles size(ft^2) and number of bedrooms
#---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from pathlib import Path


#-------------FUNCTIONS-------------#
#cost error function through least squares
def computeCost(X, y, theta): 
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

#linear regression through GD  function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        
        theta = temp
        cost[i] = computeCost(X, y, theta)
    
    return theta, cost

#-------------VARIABLES-------------#
path = Path('/home/beehive/Documents/ML-SelfStudy/Linear-Regression/ex1data2.txt')
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# conduct feature normalization to prevent the prediction from being weighted to far one way:
#subtract from each value in the feature the mean of that feature the divide by the standard deviation. 
data = (data - data.mean()) / data.std()
data.head

# add ones column.
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X2 = data.iloc[:,0:cols-1]
y2 = data.iloc[:,cols-1:cols]

# convert to matrices and initialize theta. 
X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

#alpha is the learning rate; helps determine how quickly the algorith will converge to the optimal solution
#iters is the number of steps (iterations)
alpha = 0.01
iters = 1000

# linear regression oin the dataset through GD 
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)

# the cost (error) of the model
cost_error = computeCost(X2, y2, g2)

#using sklearn, simply get the linear regression task completed
model = linear_model.LinearRegression()
sk_simple_lr = model.fit(X2,y2)


#-------------PLOT-------------#
#plot thge training progress, confirming the error was decreasing with each iteration.
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


#-------------INITIATE-------------#
if __name__ == '__main__': 
    #print("Initiate")
    #print(data)\
    print(cost_error)
    print(g2)
    print(sk_simple_lr)
    
