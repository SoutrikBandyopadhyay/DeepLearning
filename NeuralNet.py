# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 17:32:02 2019

@author: Soutrik
"""


def sigmoid(x,derivative=False):
    if(not derivative):
        return 1/(1+np.exp(-x))
    else:
        return np.exp(x)/((np.exp(x) + 1 )**2)

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

def crossentropy(prediction,desired):
    return -sum(np.multiply(desired,np.log(prediction)))[0]


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
    
    def updateWeightsAndBias(self,learningRate):
        self.weights -= learningRate*self.weightGrad
        self.bias -= learningRate*self.biasGrad

    def getActivation(self):
        return self.activation
    
    def forward(self,prevLayer):
        x = prevLayer.getActivation()
        
        self.preActivation = np.add(np.matmul(self.weights,x),self.bias)
        self.activation = self.transferFunc(self.preActivation)
        


    def backProp(self,prevLayer,nextLayer,learningRate=1):
        if(self.transferFunc == sigmoid):
            temp = sigmoid(self.preActivation,derivative=True)
            self.gradUptoThisLayer = np.multiply(nextLayer.gradToSendBack,temp)
                
        self.gradToSendBack = np.matmul(np.transpose(self.weights),self.gradUptoThisLayer)
        self.weightGrad = np.matmul(self.gradUptoThisLayer,np.transpose(prevLayer.activation))
        self.biasGrad = self.gradUptoThisLayer
        
        self.updateWeightsAndBias(learningRate)
        
    def backPropForLastLayer(self,prediction,desired,prevLayer,learningRate=1):
        if(self.transferFunc == softmax):
           
            self.gradUptoThisLayer = np.subtract(prediction,desired)
            
            self.gradToSendBack = np.matmul(np.transpose(self.weights),self.gradUptoThisLayer)

            self.weightGrad = np.matmul(self.gradUptoThisLayer,np.transpose(prevLayer.activation))
            self.biasGrad = self.gradUptoThisLayer
            
            self.updateWeightsAndBias(learningRate)


class Input:
    def __init__(self,nodes):
        self.nodes = nodes
        self.activation = np.zeros((nodes,1))
        
    def forward(self,x):
        self.activation = x

        
    def getActivation(self):
        return self.activation


class Model:
    def __init__(self,learningRate=0.2,lossFunction = crossentropy):
        self.layerList = []
        self.learningRate = learningRate
        self.lossFunction = lossFunction
    
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
     
    def backPropagate(self,prediction,desired):
        
        self.layerList[-1].backPropForLastLayer(prediction,desired,self.layerList[-2],self.learningRate)
        
        for i in range(len(self.layerList)-2,0,-1):
            nextLayer = self.layerList[i+1]
            prevLayer = self.layerList[i-1]
            
            self.layerList[i].backProp(prevLayer,nextLayer,self.learningRate)
        
        
        
        
        
#prediction = self.feedForward(x)

    def fitOnOne(self,x,desired):
        x = np.transpose(np.array([x]))
        desired = np.transpose(np.array([desired]))
        
        prediction = self.feedForward(x)
        loss = self.lossFunction(prediction,desired)
        self.backPropagate(prediction,desired)
        return loss
    
    def evaluateOnOne(self,x,desired):
        x = np.transpose(np.array([x]))
        desired = np.transpose(np.array([desired]))
        
        prediction = self.feedForward(x)
        loss = self.lossFunction(prediction,desired)
        
        return loss
    
    def fitOnBatch(self,Xs,Ys):
        sumLoss = 0
        count = 0
        for x,y in zip(Xs,Ys):
            loss = model.fitOnOne(x,y)
            sumLoss += loss
            count+=1
            
        return sumLoss/count
    
    def evaluateOnBatch(self,Xs,Ys):
        sumLoss = 0
        count = 0
        for x,y in zip(Xs,Ys):
            loss = model.evaluateOnOne(x,y)
            sumLoss += loss
            count+=1
            
        return sumLoss/count
    
    
    def fit(self,xTrain,yTrain,epochs,validationData=None):

        history = {}
        lossData = []
        valLossData = []
        for i in range(epochs):
            
            loss = self.fitOnBatch(xTrain,yTrain)
            lossData.append(loss)
            print("\n\n ____________ Epoch {} _______________".format(i))
            print("\n Loss = {}".format(loss))
            
            
            
            if(validationData):
                valLoss = self.evaluateOnBatch(validationData[0],validationData[1])
                valLossData.append(valLoss)
                print("\n Validation Loss = {}".format(valLoss))
                
        
        history['loss'] = np.array(lossData)
        history['valLoss'] = np.array(valLossData)
        
        return history
                

if(__name__=="__main__"):
    import numpy as np
    import matplotlib.pyplot as plt

    model = Model(learningRate=0.2)
    model.add(Input(2))
    model.add(Layer(5))
    model.add(Layer(5))
    model.add(Layer(5))
    model.add(Layer(2,activation=softmax))
    
    X = np.array([
         [0.1,0.1],
         [0.2,0.2],
         [0.3,0.3]
    ])
        
    Y = np.array([
         [1,0],
         [0,1],
         [1,0]
    ])
    
    
    
    
        
    history = model.fit(X,Y,epochs=20)
    
    plt.plot(history['loss'])
    plt.show()
    
