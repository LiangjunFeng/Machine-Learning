#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : naive_bayes_classifier.py
# Date   : 2017/09/21 1:29
# Version: 0.1
# Description: naive_bayes_classifier
 
from pandas import read_csv
import numpy

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
    mid = int(len(data)*0.6)    
    trainData = data[0:mid]     #set the top 60% as train data   
    testData = data[mid:]       #set the left as tet data
    
    trainLabel = label[0:mid,0]
    testLabel = label[mid:,0]
    
    return trainData,trainLabel,testData,testLabel  #return train data,train label,test data,test label

def featureIterator(Data):                          # the iterator of feature's data,that means return a column of matrix once  
    sampleNumbers,featureNumbers = Data.shape
    for i in range(featureNumbers):
        yield trainData[:,i]
        
def itemIterator(Data):         #the iterator of item's data，that means return a row of matrix once 
    for i in range(len(Data)):
        yield numpy.ravel(Data[i,:])

def average(feature,i,trainLabel):                 #return the average or a certain feature of a label
    sum_,count = 0,0
    for j in range(len(feature)):
        if trainLabel[j] == i:
            sum_ += feature[j]
            count += 1
    return float(sum_/count)

def variance(feature,i,trainLabel):               #return the variance or a certain feature of a label
    ave = average(feature,i,trainLabel)
    sum_,count = 0,0
    for j in range(len(feature)):
        if trainLabel[j] == i:
            sum_ += (feature[j]-ave)**2
            count += 1
    return float(sum_/count)

def gaussDistribute(x,average,variance):          #return the gaussDistribute for continuous feature
    part1 = 1/((2*numpy.pi)**0.5*(variance**0.5))
    part2 = numpy.exp(-(x-average)**2/(2*variance**2))
    return part1/part2 

def assess(result,label):                         #estimate the similarity between the predict resault and the real resault
    count = 0
    for i in range(len(result)):
        if result[i] == label[i][0]:
            count += 1
    return count/len(label)

class naive_bayes_classifier():                   #naive bayes classifier
    def __init__(self):
        self._classnumber = 0                     #recording the number of label
        self._featurenumber = 0                   #recording the number of feature
        self._samplenumber = 0                    #recording the number of traindata
        self._priorProbility = []                 #prior Probility table
        self._unionProbility = {}                 #union Probility setting with dictionary
    
    def get_priorProbility(self,Label):           #calculate the prior Probility ,applying laplacian correction
        priorProbility = [0]*self._classnumber
        for i in range(len(Label)):               #count the number of samples of every kind
            priorProbility[int(Label[i])] += 1
        for i in range(self._classnumber):        #calculate the prior Probility
            priorProbility[i] = (priorProbility[i]+1)/(len(Label)+self._classnumber) 
        self._priorProbility = priorProbility

    def get_unionProbility(self,trainData,trainLabel):     #calculate the union Probility ,applying laplacian correction
        unionProbility = {} 
        classgroup = [0]*self._classnumber        #using for recording the number of samples of every kind

        for i in range(len(trainLabel)):
            classgroup[int(trainLabel[i])] += 1   #count the number of samples of every kind

        for i in range(self._classnumber):        #creat a double loops dictionary for union Probility，the third loop will be created in next block
            unionProbility[i] = {}
            for j in range(self._featurenumber):
                unionProbility[i][j] = {}
  
        for i in range(self._classnumber):        #the kernel loop,creat the dictionary of union Probility
            for feature,j in zip(featureIterator(trainData),range(self._featurenumber)):
                if j == 0 :                       #the zeroth feature is discreted , which needs process seperetly
                    low = int(min(feature.tolist())[0])
                    high = int(max(feature.tolist())[0])
                    k = low
                    while k <= high:
                        if k not in unionProbility[i][j]:     
                            unionProbility[i][j][k] = 1     
                        for p in range(self._samplenumber):
                            if int(feature[p]) == k:
                                unionProbility[i][j][k] += 1                   
                        unionProbility[i][j][k] /= float(classgroup[i]+high-low+1)
                        k += 1
                else:
                    unionProbility[i][j][average] = average(feature,i,trainLabel)      #for the continuous features , only need to store their means and variances
                    unionProbility[i][j][variance] = variance(feature,i,trainLabel)

        self._unionProbility = unionProbility
       
    def fit(self,trainData,trainLabel):           #fitting function
        self._classnumber = max(numpy.ravel(trainLabel.tolist()))+1
        self._samplenumber,self._featurenumber = trainData.shape
        self.get_priorProbility(trainLabel)
        self.get_unionProbility(trainData,trainLabel)
    
    def predict(self,testData):                   #predict function
        predictResult = []
        res = [1]*self._classnumber
        for item in itemIterator(testData):       #calculate the probilitier of samples for different classes
            for i in range(self._classnumber):
                res[i] = self._priorProbility[i]
                for j in range(self._featurenumber):
                    if j == 0 :
                        res[i] *=  self._unionProbility[i][j][item[j]]
                    else:
                        res[i] *= gaussDistribute(item[j],self._unionProbility[i][j][average],self._unionProbility[i][j][variance])
            maxp = max(res)
            for i in range(self._classnumber):    #the kind is which own the the biggest probility
                if maxp == res[i]:
                    kind = i
                    predictResult.append(kind)
        return  predictResult


if __name__ == '__main__':
    dataSet = loadData()
    data,label = dataPrepreocess(dataSet)
    trainData,trainLabel,testData,testLabel = dataSplit(data,label)
    
    nbc = naive_bayes_classifier()
    nbc.fit(trainData,trainLabel)
    res = nbc.predict(testData)
    
    testLabel = testLabel.tolist()
    success = assess(res,testLabel)
    
    print(success)
    
        
