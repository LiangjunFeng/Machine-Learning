#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : AdaBoost.py
# Date   : 2017/09/17 11:12
# Version: 0.1
# Description: AdaBoost algorithm ,ensemble learning

import numpy
import math
from pandas import read_csv

def loadData():      #load the CSV data file
    dataSet = read_csv('/Users/zhuxiaoxiansheng/Desktop/data_set/pima-indians-diabetes/pima-indians-diabetes.csv',header = None)
    return dataSet

#data preprocess,split the atributes data and the label data,
#meanwhile filling the lack data,tranform the form of the data to matrix
def dataPrepreocess(dataSet):   
    dataSet[[0,1,2,3,4,5,6,7]] = dataSet[[0,1,2,3,4,5,6,7]].replace(0,numpy.NaN)
    data = dataSet[[0,1,2,3,4,5,6,7]]
    data.fillna(data.median(),inplace = True)              #filling the lack with median value
    label = dataSet[8]         
         
    data = numpy.matrix(data.as_matrix(columns=None))      #transfer to matrix
    label = numpy.matrix(label.as_matrix(columns=None))    #transfer to matrix 
    
    return data,label.T

def dataSplit(data,label):      #split the atributes data and the label data
    for i in range(len(label)):
        if label[i,0] == 0:
            label[i,0] = -1
    mid = int(len(data)*0.68)    
    trainData = data[0:mid]     #set the top 60% as train data   
    testData = data[mid:]       #set the left as tet data
    
    trainLabel = label[0:mid,0]
    testLabel = label[mid:,0]
    
    return trainData,trainLabel,testData,testLabel  #return train data,train label,test data,test label

class decisionRoot:                                 #use decision tree root as base classifier
    def __init__(self):                     
        self._rootAttribute = 0                     #the tree root's classification attribute
        self._thresholdValue = 0                    #the threshold value of the tree root's classification attribute
        self._minError = 0                          #the classifiers' error of classification
        self._operator = None                       #the classifier operator，the value could be ‘le’ or ‘gt’
        self._estimateLabel = None                  #the estimated label or the train set 
    
    @classmethod
    def classify(cls,data,operator,thresholdValue):          #have known the root's classification attribute,threshold value and classifier operator,classifiy the data to two classes
        estimateLabel =  numpy.ones((data.shape[0],1))
        if operator == 'le':
            estimateLabel[data <= thresholdValue] = -1       #if the operator is 'le' and if the attribute value less or equals the threshold value,set the label with -1,the else remain be 1
        else:
            estimateLabel[data > thresholdValue] = -1        #if the operator is 'gt' and if the attribute value greater than the threshold value,set the label with -1,the else remain be 1
        return estimateLabel
            
    def buildTree(self,data,label,distribute):      #derive the optimal Tree root based on the train data and distribute
        samples,attributes = data.shape
        estimateLabel = numpy.matrix(numpy.zeros((samples,1)))
        minError = numpy.inf                        #set the inital value of the minimal error as infinity
        stepNumber = 10.0                           #In order to determine the optimal classification threshold of attributes, the sample attributes are traversed from the minimum to the maximum, and the number of steps is 10
        for rootAttribute in range(attributes):
            attributeMin = data[:,rootAttribute].min()
            attributeMax = data[:,rootAttribute].max()
            stepSize = (attributeMax - attributeMin)/stepNumber    #get the step length
            for j in range(int(stepNumber+1)):      #Traversal attributes, in order to obtain the optimal classification attributes
                for operator in ['le','gt']:        #Select the operator to get the best operater
                    thresholdValue = attributeMin + j*stepSize     #the threshold value of the tree root's classification attribute
                    estimateLabel = decisionRoot.classify(data[:,rootAttribute],operator,thresholdValue)      #the label that estimated by the classifier
                    errLabel = numpy.matrix(numpy.ones((samples,1)))
                    errLabel[estimateLabel == label] = 0
                    error = float(distribute*errLabel)             #calculate the classification error
                    if error < minError:
                        minError = error                           #record parameter of the minimum error 
                        self._minError = minError
                        self._rootAttribute = rootAttribute
                        self._thresholdValue = thresholdValue
                        self._operator = operator
                        self._estimateLabel = estimateLabel.copy()
    
    def predict(self,data):                         #Input data, prediction results
        samples = data.shape[0]
        result =  numpy.matrix(decisionRoot.classify(data[:,self._rootAttribute],self._operator,self._thresholdValue))
        return result
        
        
class AdaBoost:  
    def __init__(self):              
        self._rootNumber = 0               #the root's number that need to train
        self._decisionRootArray = []       #store the root that have been trained
        self._alphaList = []               #record the alpha value of every root
    
    @classmethod
    def exp(cls,vector):                   #using exp function process all the items in the vector
        for i in range(len(vector)):
            vector[i,0] = math.exp(vector[i,0])
        return vector
          
    def trainRoot(self,data,label,rootNumber = 45):      #train root,and store them
        samples,attributes = data.shape
        self._rootNumber = rootNumber
        distribute = numpy.matrix(numpy.ones((1,samples))/float(samples))
        for i in range(self._rootNumber):               #numbers of roots be trained
            root = decisionRoot()
            root.buildTree(data,label,distribute)       #make and set up a root
            alpha = 0.5 * math.log((1.0 - root._minError)/max(root._minError,1e-16))
            self._alphaList.append(alpha)
            expValue = AdaBoost.exp(numpy.multiply(-1*alpha*label,root._estimateLabel))
            distribute = numpy.multiply(distribute,expValue.T)
            distribute = distribute/float(distribute.sum())            #refresh the distribute
            self._decisionRootArray.append(root)
    
    def vote(self,predictMatrix):          #using Voting rules determines classification results
        samples = predictMatrix[0].shape[0]
        result = []
        for i in range(samples):
            sign = 0
            for j in range(self._rootNumber):
                sign += predictMatrix[j][i,0] * self._alphaList[j]
            if sign >= 0:
                result.append(1)        
            else:
                result.append(-1)           #record the result of the voting
        return result
                
    def predict(self,data):                #Integrated voting results are based on many well trained classifiers
        predictMatrix = []
        for i in range(self._rootNumber):  #Each classifier is traversed to obtain a classification value
            root = self._decisionRootArray[i]
            result = root.predict(data)
            predictMatrix.append(result)
        return numpy.matrix(self.vote(predictMatrix)).T       #Returns the voting results for each classifier

def assess(predictResult,actualLabel):      #Evaluate classifier accuracy
    count = 0
    for i in range(predictResult.shape[0]):
        if predictResult[i,0] == actualLabel[i,0]:
            count += 1
    return count/ float(predictResult.shape[0])
            
if __name__ == '__main__': 
    dataSet = loadData()
    data,label = dataPrepreocess(dataSet)
    trainData,trainLabel,testData,testLabel = dataSplit(data,label)     #Training set and test set are obtained
    
    Ada = AdaBoost()           
    Ada.trainRoot(trainData,trainLabel)
    result = Ada.predict(testData)
    print(assess(result,testLabel))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
