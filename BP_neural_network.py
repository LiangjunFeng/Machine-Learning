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
def rand(low,high):   #随机函数
    return (high-low)*random.random()+low

def sigmoid(x):       #sigmoid函数
    return 1.0/(1+math.exp(-x))

def vectorSigmoid(vect):     #对一个向量里的所有值进行sigmoid函数处理
    length = numpy.shape(vect)[1]
    for i in range(length):
        vect[0,i] = sigmoid(vect[0,i])
    return vect

def makeMatrix(length,bordth,fill = 0.0):    #创建矩阵
    mat = []
    for  i in range(length):
        mat.append([fill]*bordth)
    return numpy.mat(mat)

def randfillMatrix(mat,low,high):      #矩阵值的随机填充
    length = numpy.shape(mat)[0]
    bordth = numpy.shape(mat)[1]
    for i in range(length):
        for j in range(bordth):
            mat[i,j] = rand(low,high)
    return mat

def vectorMultiply(vect1,vect2):       #向量相乘得到一个矩阵
    length = numpy.shape(vect1)[1]
    bordth = numpy.shape(vect2)[1]
    mat = makeMatrix(length,bordth)
    for i in range(length):
        for j in range(bordth):
            mat[i,j] = vect1[0,i]*vect2[0,j]
    return mat

class BPNeuralNetwork:               
    def __init__(self,inputNodes,hiddenNodes,outputNodes):    #初始化变量
        self._inputNodes = inputNodes      #输入层结点个数
        self._hiddenNodes = hiddenNodes    #隐藏层结点个数
        self._outputNodes = outputNodes    #输出层结点个数
        self._inputLayer = []              #输入层数据
        self._hiddenLayer = []             #隐藏层数据
        self._outputLayer = []             #输出层数据
        self._inputWeights = []            #输入层到隐藏层连接权
        self._outputWeights = []           #隐藏层到输出层连接权
        self._hiddenThreshold = []         #隐藏层激活阈值
        self._outputThreshold = []         #输出层阈值
    
    def setup(self):        
        self._inputLayer = makeMatrix(1,self._inputNodes,1)
        self._hiddenLayer = makeMatrix(1,self._hiddenNodes,1)
        self._outputLayer = makeMatrix(1,self._outputNodes,1)    #三层网络架构初始赋值
        
        self._hiddenThreshold = makeMatrix(1,self._hiddenNodes,1)
        self._outputThreshold = makeMatrix(1,self._outputNodes,1)    #两层激活阈值初始赋值
        
        self._inputWeights = makeMatrix(self._inputNodes,self._hiddenNodes)
        self._outputWeights = makeMatrix(self._hiddenNodes,self._outputNodes)  #创建权重矩阵
        
        self._inputWeights = randfillMatrix(self._inputWeights,-1,1)
        self._outputWeights = randfillMatrix(self._outputWeights,-1,1)    #权重矩阵初始随机幅值
                 
    def getInput(self,_input):                #将一个输入赋值到输入层
        for i in range(self._inputNodes):
            self._inputLayer[0,i] = _input[i]
    
    def predict(self,_input):                 #根据输入输出预测值到输出层
        self.getInput(_input)
        self._hiddenLayer = vectorSigmoid(self._inputLayer*self._inputWeights - self._hiddenThreshold)
        self._outputLayer = vectorSigmoid(self._hiddenLayer*self._outputWeights - self._outputThreshold)
                    
    def backPropagation(self,y,lRate):       #反响传播更新激活阈值与连接权    
        G = numpy.multiply(numpy.multiply(self._outputLayer,1-self._outputLayer),y-self._outputLayer)
        self._outputWeights = self._outputWeights + lRate*vectorMultiply(self._hiddenLayer,G)
        self._outputThreshold = self._outputThreshold - lRate*G
        E = numpy.multiply(numpy.multiply(self._hiddenLayer,1-self._hiddenLayer),(self._outputWeights*G.T).T)
        self._inputWeights = self._inputWeights + lRate*vectorMultiply(self._inputLayer,E)
        self._hiddenThreshold = self._hiddenThreshold - lRate*E
        return float((y-self._outputLayer)*(y-self._outputLayer).T) 
    
    def fit(self,input_,y_,limit = 10000,accuracy = 0.001,lRate = 0.2):     #拟合数据
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
                    
if __name__ == '__main__':           #以异或运算举例
    nn = BPNeuralNetwork(2,7,1)        
    cases = [[0,0],[0,1],[1,0],[1,1]]
    labels = [[0],[1],[1],[0]]        
    nn.fit(cases,labels)
    for case in cases:
        nn.predict(case)
        print(case,'  ',nn._outputLayer)
        
        
        
        
        
        
        
        
        
        
        