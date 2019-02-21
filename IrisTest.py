# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:35:02 2019

@author: Parth
"""

from NeuralNet import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("Iris.csv")

def processData(df):
    X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
    Y = df['Species']
    Y = pd.get_dummies(Y)
    X = X/10
    return X.values,Y.values


X,Y = processData(df)    

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1, random_state=42)


model = Model(learningRate=0.3)

model.add(Input(4))
model.add(Layer(5))
model.add(Layer(6))
model.add(Layer(6))
model.add(Layer(3,activation=softmax))

history = model.fit(xTrain,yTrain,epochs=1000,validationData=[xTest,yTest])

plt.plot(history['loss'],label="Loss")
plt.plot(history['valLoss'],label="valLoss")
plt.legend()
plt.show()