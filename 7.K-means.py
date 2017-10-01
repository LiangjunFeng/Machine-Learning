#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : K-means.py 
# Date   : 2017/09/29 3:25
# Version: 0.1
# Description: K-means algorithm ,clustering

import numpy
import random
import matplotlib.pyplot as plt 

def creatData():
    data1 = numpy.mat(numpy.random.randint(0,12,size=(2,50)))
    data2 = numpy.mat(numpy.random.randint(8,20,size=(2,50)))    #creat two different kinds of data
    data = numpy.hstack((data1,data2)).T                         #combine the matrix
      
    label1 = numpy.mat(numpy.zeros(50)).T
    label2 = numpy.mat(numpy.ones(50)).T                         #create the label
    label = numpy.vstack((label1,label2))                        #combine the matrix
    return data,label
    
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
        

if __name__ == '__main__':    
    data,label = creatData()
    
    C = K_means()
    C.clustering(data,2)
    
    for i in range(len(C._classSetList[0])):              #visualable
        p1=plt.scatter(data[C._classSetList[0][i],0],data[C._classSetList[0][i],1],marker='x',color='g') 
    for i in range(len(C._classSetList[1])):
        p2=plt.scatter(data[C._classSetList[1][i],0],data[C._classSetList[1][i],1],marker='+',color='r')
    plt.show()
    
    
