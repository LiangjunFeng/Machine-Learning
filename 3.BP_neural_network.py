#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : BP_neural_network.py
# Date   : 2017/09/17 11:12
# Version: 0.1
# Description: BP neural network

import random
import math
import numpy

random.seed(0)
def rand(low,high):   #random function
    return (high-low)*random.random()+low

def sigmoid(x):       #sigmoid function
    return 1.0/(1+math.exp(-x))

def vectorSigmoid(vect):     #using sigmoid function processes all the elements in the vector
    length = numpy.shape(vect)[1]
    for i in range(length):
        vect[0,i] = sigmoid(vect[0,i])
    return vect

def makeMatrix(length,bordth,fill = 0.0):    #creat a matrix
    mat = []
    for  i in range(length):
        mat.append([fill]*bordth)
    return numpy.mat(mat)

def randfillMatrix(mat,low,high):      #filling the matrix with random number
    length = numpy.shape(mat)[0]
    bordth = numpy.shape(mat)[1]
    for i in range(length):
        for j in range(bordth):
            mat[i,j] = rand(low,high)
    return mat

def vectorMultiply(vect1,vect2):       #vector x vector to get a matrix
    length = numpy.shape(vect1)[1]
    bordth = numpy.shape(vect2)[1]
    mat = makeMatrix(length,bordth)
    for i in range(length):
        for j in range(bordth):
            mat[i,j] = vect1[0,i]*vect2[0,j]
    return mat

class BPNeuralNetwork:               
    def __init__(self,inputNodes,hiddenNodes,outputNodes):    #init the variables
        self._inputNodes = inputNodes      #inputlayer's nodes number
        self._hiddenNodes = hiddenNodes    #hiddenlayer's nodes number
        self._outputNodes = outputNodes    #outputlayer's nodes number
        self._inputLayer = []              #inputlayer's data 
        self._hiddenLayer = []             #hiddenlayer's data
        self._outputLayer = []             #outputlayer's data
        self._inputWeights = []            #the weights from inputlayer to hiddenlayer
        self._outputWeights = []           #the weighyts from hiddenlayer to outputlayer
        self._hiddenThreshold = []         #the hiddenlayer's Threshold
        self._outputThreshold = []         #the outputlayer's Threshold
    
    def setup(self):        
        self._inputLayer = makeMatrix(1,self._inputNodes,1)
        self._hiddenLayer = makeMatrix(1,self._hiddenNodes,1)
        self._outputLayer = makeMatrix(1,self._outputNodes,1)    #give inital data to three layers's structure
        
        self._hiddenThreshold = makeMatrix(1,self._hiddenNodes,1)
        self._outputThreshold = makeMatrix(1,self._outputNodes,1)    #give inital data to two layer
        
        self._inputWeights = makeMatrix(self._inputNodes,self._hiddenNodes)
        self._outputWeights = makeMatrix(self._hiddenNodes,self._outputNodes)  #creat weights' matrix
        
        self._inputWeights = randfillMatrix(self._inputWeights,-1,1)
        self._outputWeights = randfillMatrix(self._outputWeights,-1,1)    #give inital random data to weights' matrix
                 
    def getInput(self,_input):                #load an example to inputlayer
        for i in range(self._inputNodes):
            self._inputLayer[0,i] = _input[i]
    
    def predict(self,_input):                 #predict the outputlayer according to the input data
        self.getInput(_input)
        self._hiddenLayer = vectorSigmoid(self._inputLayer*self._inputWeights - self._hiddenThreshold)
        self._outputLayer = vectorSigmoid(self._hiddenLayer*self._outputWeights - self._outputThreshold)
                    
    def backPropagation(self,y,lRate):       #using backPropagation to reset the weights and threshold   
        G = numpy.multiply(numpy.multiply(self._outputLayer,1-self._outputLayer),y-self._outputLayer)
        self._outputWeights = self._outputWeights + lRate*vectorMultiply(self._hiddenLayer,G)
        self._outputThreshold = self._outputThreshold - lRate*G
        E = numpy.multiply(numpy.multiply(self._hiddenLayer,1-self._hiddenLayer),(self._outputWeights*G.T).T)
        self._inputWeights = self._inputWeights + lRate*vectorMultiply(self._inputLayer,E)
        self._hiddenThreshold = self._hiddenThreshold - lRate*E
        return float((y-self._outputLayer)*(y-self._outputLayer).T) 
    
    def fit(self,input_,y_,limit = 10000,accuracy = 0.001,lRate = 0.2):     #using data fit the model
        self.setup()
        num = len(input_)
        error,times = num,1
        while((error > accuracy) and (times < limit)):
            i,error = 0,0 
            while(i < num):
                _input = input_[i]
                y = y_[i]
                self.predict(_input)
                error += self.backPropagation(y,lRate)               
                i += 1
            times += 1 
                    
if __name__ == '__main__':           #taking exclusive_OR as an example
    nn = BPNeuralNetwork(2,7,1)        
    cases = [[0,0],[0,1],[1,0],[1,1]]
    labels = [[0],[1],[1],[0]]        
    nn.fit(cases,labels)
    for case in cases:
        nn.predict(case)
        print(case,'  ',nn._outputLayer)
        
        
        
        
        
        
        
        
        
        
        
