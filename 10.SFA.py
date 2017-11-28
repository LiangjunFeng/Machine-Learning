#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : SFA.py
# Date   : 2017/011/28 10:00
# Version: 0.1
# Description: Slowness Feature Analysis,Apply it to reduce pictures' dimension

import skimage.io as io
import numpy as np
import math
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

def LoadData(number):                                     #Load the picture data
    if number == 1:
        path = '/Users/zhuxiaoxiansheng/Desktop/yale_faces/*.bmp'        #the data's path
        num =11
    elif number == 2:
        path = '/Users/zhuxiaoxiansheng/Desktop/orl_faces_full/*.pgm'    #the data's path
        num =10
    pictures = io.ImageCollection(path)
    data = []
    for i in range(len(pictures)):
        data.append(np.ravel(pictures[i].reshape((1,pictures[i].shape[0]*pictures[i].shape[1]))))
    label = []
    for i in range(len(data)):
        label.append(int(i/num))
    return np.matrix(data),np.matrix(label).T   

def SplitData(data,label,number,propotion):               #split data to train data set and test data set 
    if number == 1:
        classes = 15
    elif number == 2:
        classes = 40
    samples = data.shape[0]
    perClass = int(samples/classes)
    selected = int(perClass*propotion)
        
    trainData,testData = [],[]
    trainLabel,testLabel = [],[]
    count1 = []
    for i in range(classes):
        count2,k = [],math.inf
        for j in range(selected):
            count2.append(k)
            k = random.randint(0,perClass-1)
            while k in count2:
                k = random.randint(0,perClass-1)
            trainData.append(np.ravel(data[perClass*i+k]))
            trainLabel.append(np.ravel(label[perClass*i+k]))
            count1.append(11*i+k)
    for i in range(samples):
        if i not in count1:
            testData.append(np.ravel(data[i]))
            testLabel.append(np.ravel(label[i]))
    return np.matrix(trainData),np.matrix(trainLabel),np.matrix(testData),np.matrix(testLabel)   

class SFA:                                                #slow feature analysis class
    def __init__(self):
        self._Z = []
        self._B = []
        self._eigenVector = []
        
    def getB(self,data):
        self._B = np.matrix(data.T.dot(data))/(data.shape[0]-1)
        
    def getZ(self,data):
        derivativeData = self.makeDiff(data)
        self._Z = np.matrix(derivativeData.T.dot(derivativeData))/(derivativeData.shape[0]-1)    
    
    def makeDiff(self,data):
        diffData = np.mat(np.zeros((data.shape[0],data.shape[1])))
        for i in range(data.shape[1]-1):
            diffData[:,i] = data[:,i] - data[:,i+1]
        diffData[:,-1] = data[:,-1] - data[:,0]
        return np.mat(diffData)         
        
    def fit_transform(self,data,threshold = 1e-7,conponents = -1):
        if conponents == -1:
            conponents = data.shape[0]
        self.getB(data)
        U,s,V = np.linalg.svd(self._B)

        count = len(s)
        for i in range(len(s)):
            if s[i]**(0.5) < threshold:
                count = i
                break
        s = s[0:count]
        s = s**0.5    
        S = (np.mat(np.diag(s))).I
        U = U[:,0:count]
        whiten = S*U.T            
        Z = (whiten*data.T).T
        
        self.getZ(Z)
        PT,O,P = np.linalg.svd(self._Z)

        self._eigenVector = P*whiten
        self._eigenVector = self._eigenVector[-1*conponents:,:]
        
        return data.dot(self._eigenVector.T)
        
    def transfer(self,data):
        return data.dot(self._eigenVector.T)
    
def show_accuracy(predictLabel,Label):                            #show the accuracy of the classifier
    Label = np.ravel(Label).tolist()
    predictLabel = predictLabel.tolist()
    count = 0
    for i in range(len(Label)):
        if Label[i] == predictLabel[i]:
            count += 1
    print(count/len(Label))
    

def Faceidentifier( trainDataSimplified,trainLabel,testDataSimplified,testLabel):     #three different kinds of classifers
    print("=====================================")    
    print("GaussianNB")
    clf1 = GaussianNB()
    clf1.fit(trainDataSimplified,np.ravel(trainLabel))
    predictTestLabel1 = clf1.predict(testDataSimplified)
    show_accuracy(predictTestLabel1,testLabel)
    print()
    
    print("SVC")
    clf3 = SVC(C=8.0)
    clf3.fit(trainDataSimplified,np.ravel(trainLabel))
    predictTestLabel3 = clf3.predict(testDataSimplified)
    show_accuracy(predictTestLabel3,testLabel)
    print()
    
    print("LogisticRegression")
    clf4 = LogisticRegression()
    clf4.fit(trainDataSimplified,np.ravel(trainLabel))
    predictTestLabel4 = clf4.predict(testDataSimplified)
    show_accuracy(predictTestLabel4,testLabel)
    print()
    print("=====================================") 

if __name__ == "__main__":

    data,label = LoadData(2)
    pca = PCA(30,True,True)
    datapca = pca.fit_transform(data) 
    trainData,trainLabel,testData,testLabel = SplitData(datapca,label,2,0.6)  
    
    trainData = preprocessing.scale(trainData.T).T
    testData = preprocessing.scale(testData.T).T   
    
    sfa = SFA()
    trainDataS = sfa.fit_transform(trainData,conponents = 30)
    testDataS = sfa.transfer(testData) 

    Faceidentifier(trainDataS,trainLabel,testDataS,testLabel)           
    

    
    
    
    
    
    
    
    
    

