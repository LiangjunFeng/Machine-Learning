#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : pca.py
# Date   : 2017/010/1 6:25
# Version: 0.1
# Description: pca algorithm ,use k-means algorithm for clustering

from pandas import read_csv
import numpy
import random
import matplotlib.pyplot as plt 

def loadData():      # read the CSV data file
    dataSet = read_csv('/Users/zhuxiaoxiansheng/Desktop/data_set/Epileptic Seizure Recognition Data Set/data.csv',header = None)
    return dataSet

def dataPrepreocess(dataSet):   #preprocess the data,separating the attributes and label
    data = dataSet.iloc[:,0:178]
    label = dataSet[178]         
         
    data = numpy.matrix(data.as_matrix(columns=None))      #convert to matrix
    data = data[0:5000,:]
    label = numpy.matrix(label.as_matrix(columns=None))    #convert to matrix
    label = label[:,0:5000]
    
    return data,label.T


def zeroMean(data):                        #minus the mean value from the data
    meanValue = numpy.mean(data,axis = 0)
    meanData = data - meanValue
    return meanData,meanValue

def percentage(eigenValues,per):           #remain the 'percentage' of the data
    sortArray = numpy.sort(eigenValues)[-1::-1]
    arraySum = sum(sortArray)
    tempSum,num = 0,0
    for i in sortArray:
        tempSum += i
        num += 1
        if tempSum >= arraySum * per:
            return num

def pca(data,per = 0.95):                    #the pca function
    data,mean = zeroMean(data)
    covariance = numpy.cov(data,rowvar = 0)
    eigenValues,eigenVectors = numpy.linalg.eig(numpy.mat(covariance))      calculate the eigenvalue
    num = percentage(eigenValues,per)
    eigenVlueIndice = numpy.argsort(eigenValues)
    sel_eigenValInd = eigenVlueIndice[-1:-(num+1):-1]
    sel_eigenVectors = eigenVectors[:,sel_eigenValInd]
    lowerData = data*sel_eigenVectors        #the lower dimension data
    recondata = (lowerData*sel_eigenVectors.T)+mean
    return lowerData,recondata

def minkowskiDistance(vector1,vector2,p=2):                      #calculate the distance between two samples
    vector1 = numpy.ravel(vector1)
    vector2 = numpy.ravel(vector2)
    length,distance = len(vector1),0
    for i in range(length):
        distance += abs(vector1[i] - vector2[i])**p
    return distance**(1.0/p)

def randomInt(minimum,maximum):                                  #use to choose inital random averageVector
    return random.randrange(minimum,maximum,1)

class K_means:                                #K-means class
    def __init__(self):
        self._classes = 0                     #record the number of the class that want to cluster from the data      
        self._averageVectorList = []          #the average Vector list
        self._classSetList = []               #the list of different kinds data
    
    def setUp(self,data,number):              #setup the K-means class
        self._classes = number
        self._averageVectorList = [0]*self._classes
        for i in range(self._classes):
            self._classSetList.append([])
    
    def randomVector(self,data):              #init the beginning average Vector
        for i in range(self._classes):
            self._averageVectorList[i] = data[randomInt(0,data.shape[0]),:]
            
    def classifiy(self,data):                 #compare distance to classifiy the samples
        samples,attributes = data.shape
        for i in range(samples):
            k = 0
            for j in range(self._classes):
                if minkowskiDistance(data[i,:],self._averageVectorList[k]) > minkowskiDistance(data[i,:],self._averageVectorList[j]):
                    k = j
            self._classSetList[k].append(i)
    
    def renewAverageVector(self,data):       #renew the average vector
        for i in range(self._classes):
            sum = 0
            for j in range(len(self._classSetList[i])):
                sum += data[self._classSetList[i][j],:]
            self._averageVectorList[i] = numpy.ravel(sum/len(self._classSetList[i]))
            
    def compare(self,vector):                #campare the new average vector with the old average vector,if  they are equal,return True 
        if vector == []:
            return False
        for i in range(len(vector)):
            if not (self._averageVectorList[i] == vector[i]).all():
                    return False
        return True
           
    def clustering(self,data,number,limits = 100):
        self.setUp(data,number)
        self.randomVector(data)
        oldAverageVector,times = [],0
        while (not self.compare(oldAverageVector)) and (times < limits):  #set the limit times
            oldAverageVector = self._averageVectorList.copy()
            self._classSetList.clear()
            for i in range(self._classes):
                self._classSetList.append([])
            self.classifiy(data)
            self.renewAverageVector(data)
            times += 1
                   
    def assess(self,label):                  #using for assess the result,but it's just for reference, not accuracy
        count1 = [0]*int(max(label))
        count2 = [0]*self._classes
        for i in range(label.shape[0]):
            count1[int(label[i]-1)] += 1
        for i in range(label.shape[0]):
            for j in range(self._classes):
                count2[j] = len(self._classSetList[j])
        count1.sort()
        count2.sort()
        return count1,count2
        
if __name__ ==  '__main__':
    dataSet = loadData()
    data,label = dataPrepreocess(dataSet)
    lowerData,recondata = pca(data,0.95)
    
    C = K_means()
    C.clustering(lowerData,2)
    count1,count2 = C.assess(label)
    print('Truth: ', count1)
    print('Clustering: ', count2)
    
    for i in range(len(C._classSetList[0])):     #visualization
        p1=plt.scatter(data[C._classSetList[0][i],1],data[C._classSetList[0][i],6],marker='x',color='g') 
    for i in range(len(C._classSetList[1])):
        p2=plt.scatter(data[C._classSetList[1][i],1],data[C._classSetList[1][i],6],marker='*',color='r')
    plt.show()
    
    
