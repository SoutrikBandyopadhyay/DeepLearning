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
        
        print("\nAct = {}".format(self.activation))


    
    def backprop(self,nextLayer):
        
        pass
    
    def backPropForLastLayer(self,prediction,desired,prevLayer):
        if(self.transferFunc == softmax):
            #Must be One Hot Vector
            print(prediction.shape)
            print(desired.shape)

            
            self.gradUptoThisLayer = np.subtract(prediction,desired)
            
            self.gradToSendBack = np.matmul(np.transpose(self.weights),self.gradUptoThisLayer)
            
            print("\nUpto this layer {}".format(self.gradUptoThisLayer))
            print("\nTo send back {}".format(self.gradToSendBack))
        

            self.weightGrad = np.matmul(self.gradUptoThisLayer,np.transpose(prevLayer.activation))
            self.biasGrad = self.gradUptoThisLayer
            
            print("\nWeight Grad {}".format(self.weightGrad))



















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
     
    def backPropagate(self,x,desired):
        prediction = self.feedForward(x)
        self.layerList[-1].backPropForLastLayer(prediction,np.transpose(np.array([desired])),self.layerList[-2])
        
        
        print("\nPrediction = {}".format(prediction))
        print("\nDesired = {}".format(desired))
        
        
    
model = Model()
model.add(Input(1))
model.add(Layer(3))
model.add(Layer(2,activation=softmax))

model.backPropagate([0.1],[1,0])

        
        
