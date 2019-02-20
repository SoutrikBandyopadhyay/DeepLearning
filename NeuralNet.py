# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:32:02 2019

@author: Soutrik
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x,derivative=False):
    if(not derivative):
        return 1/(1+np.exp(-x))

def softmax(Y):
    Y = np.array(list(map(lambda x: np.exp(x),Y)))
    Y = np.array(list(map(lambda x: x/sum(Y),Y)))    
    return Y

def relu(x,derivative=False):
    if(not derivative):
        if(x<=0):
            return 0
        else:
            return x

class Layer:
    def __init__(self,nodes,activation=sigmoid):
        self.nodes = nodes    
        self.transferFunc = activation
        
    def initialize(self):
        self.weights = np.random.rand(self.nodes,self.prevNodes)
        self.bias = np.random.rand(self.nodes,1)
        #ai
        self.preActivation = np.zeros((self.nodes,1))
        #hi
        self.activation = np.zeros((self.nodes,1))
        
        #BackPropvar
        self.gradUptoThisLayer = np.zeros((self.nodes,1))
        self.gradToSendBack = np.zeros((self.prevNodes,1))
        
    def getWeightsandBias(self):
        return self.weights,self.bias
    
    def getActivation(self):
        return self.activation
    
    def forward(self,prevLayer):
        x = prevLayer.getActivation()
        
        self.preActivation = np.add(np.matmul(self.weights,x),self.bias)
        self.activation = self.transferFunc(self.preActivation)
        

class Input:
    def __init__(self,nodes):
        self.nodes = nodes
        self.activation = np.zeros((nodes,1))
        
    def forward(self,x):
        self.activation = np.transpose(np.array([x]))

        
    def getActivation(self):
        return self.activation


class Model:
    def __init__(self):
        self.layerList = []
    
    def add(self,layer):
        if(len(self.layerList)):
            layer.prevNodes = self.layerList[-1].nodes
            layer.initialize()
            
        self.layerList.append(layer)
    
    def feedForward(self,x):
        self.layerList[0].forward(x)
        for i in range(1,len(self.layerList)):
            j = i-1
            prevLayer = self.layerList[j]
            self.layerList[i].forward(prevLayer)
            
        return self.layerList[-1].activation
        
    
model = Model()
model.add(Input(3))
model.add(Layer(15))
model.add(Layer(15))
model.add(Layer(15))
model.add(Layer(5,activation=softmax))


y = model.feedForward([0.1,0.2,0.5])

        
        
