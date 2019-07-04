#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 22:13:11 2019

@author: beebrain
@original : johnwittenauer.net & curious insight 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("/home/beebrain/Documents/ML-SelfStudy/Linear-Regression/ex1data1.txt")
data = pd.read_csv(path, header=None, names=['Population','Profit'])
data.head()

#.describe describes the data and .plot plots it
scribe_data = data.describe()
#plt_data = data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))

# defining the Cost Function, using LeastSquares
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T)-y), 2)
    return np.sum(inner) / (2 * len(X))

# append a ones column to the front of the data set
data.insert(0,'Ones',1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# preform the gadient descent on the theta parameters (basically the derivative)
#Compute the gradient of the error term in order to figure out the appropriate direction
#to move the parameter vector.
def gradientDescent(X, y, theta, alpha, iters): 
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters): 
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) *np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost

# initialize variables for learning rate and iterations
#alpha is the learning rate; helps determine how quickly the algorith will converge to the optimal solution
#iters is the number of steps (iterations)
alpha = 0.01
iters = 1000

# perform gradient descent to fit the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

#view the solution of the GD using .linspace and then evaluate the points
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

#gradient descent also outputs a vector with the cost at each iteration. Plot below
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


if __name__ == '__main__':
    #print(data)
    #print(scribe_data)
    #print(plt_data)
    print(X.shape, theta.shape, y.shape)
    print(computeCost(X, y, theta))
    print(g)
    #new cost with use of GD outside of 'least squares/cost function'
    print(computeCost(X, y, g))