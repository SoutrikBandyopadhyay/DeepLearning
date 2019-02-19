# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:24:03 2019

@author: Soutrik
"""
import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.6,0.8])
Y = np.array([0.9,0.5])

def sigmoid(w,b,x):
    return 1/(1+np.exp(-(w*x+b)))

def error(w,b):
    error = 0
    for x,y in zip(X,Y):
        pred = sigmoid(w,b,x)
        error += 0.5*((pred-y)**2)
    return error


def gradB(w,b,x,y):
    pred = sigmoid(w,b,x)
    return (pred - y)* pred*(1-pred)


def gradW(w,b,x,y):
    pred = sigmoid(w,b,x)
    return (pred - y)* pred*(1-pred)*x


def trainLoop():
    w = np.random.random()
    b = np.random.random()
    learningRate = 10
    numEpochs = 1000
    
    for i in range(numEpochs):
        dW = 0
        dB = 0
        for x,y in zip(X,Y):
            dW += gradW(w,b,x,y)
            dB += gradB(w,b,x,y)

        w -= learningRate*dW
        b -= learningRate*dB
        
        print(error(w,b))
    return w,b
   
(w,b) = trainLoop()
