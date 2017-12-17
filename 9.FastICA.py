#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : FastICA.py
# Date   : 2017/10/15 6:25
# Version: 0.1
# Description: fastICA algorithm ,split voice signals

import wave
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import random
from sklearn import preprocessing 
import scipy
import scipy.io as sio 

def LoadSoundSet(path):
    filename= os.listdir(path) 
    data = []
    for i in range(len(filename)):
        f = wave.open(path+filename[i],'rb')
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        strData = f.readframes(nframes)
        waveData = np.fromstring(strData,dtype=np.int16)
        waveData = waveData*1.0/(max(abs(waveData)))

        data += waveData.tolist()
    time = np.arange(0,nframes*len(filename))*(1.0 / framerate)
    return time.tolist(),data

def LoadSound(path):
    f = wave.open(path,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)               #read the wav file
    waveData = np.fromstring(strData,dtype=np.int16) 
    waveData = waveData*1.0/(max(abs(waveData)))  #normalize the sound wave
    time = np.arange(0,nframes*nchannels)*(1.0 / framerate)
    return time.tolist(),waveData.tolist() 

def ShowRes(data):
    print("//==========================================================//")
    x = np.linspace(0,1,data.shape[1])
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(6.5, 1*data.shape[0])
    for i in range(data.shape[0]):
        axes = plt.subplot(data.shape[0],1,i+1)
        axes.set_frame_on(False) 
        axes.set_axis_off()
        plt.plot(x,data[i,:].T,color = 'black')
    plt.show()
    print("//==========================================================//")

def getRandomW(length,height):                    #make a random matrix
    W = random.random(size=(length,height))
    return W
    
def eigOrth(Data):                                #eigenormalize the data
    data = Data.copy()
    D,E = np.linalg.eig(data.dot(data.T))
    for i in range(len(D)):
        if D[i] < 1e-7:
            D[i] = 0.01
        D[i] = D[i]**0.5
    D = np.mat(np.diag(D))
    D = D.I
    data = E*D*E.T*data
    return data.real
    
def GFunction(data):                              #the first derivate function in ICA                        
    def G(x):
        y = x*math.exp(-0.5*(x**2))
        return y
    length,bordth = data.shape
    output = np.zeros((length,bordth))
    for i in range(length):
        for j in range(bordth):
            output[i,j] = G(data[i,j])
    return output

def gFunction(data):                             #the second derivate function in ICA    
    def g(x):
        y = -1*(x**2)*math.exp(-0.5*(x**2))
        return y
    length,bordth = data.shape
    output = np.zeros((length,bordth))
    for i in range(length):
        for j in range(bordth):
            output[i,j] = g(data[i,j])
    return output 

def distance(W,oldW):                            #using at judging convergence
    return abs(abs(float(W.T*oldW)) - 1)   

class ICA:                                      #ICA
    def __init__(self,conponent = -1):
        self._W = []
        self._conponent = conponent
        self._data = 0
        
    def fit_transform(self,data):
        data = preprocessing.scale(data.T)
        data = np.mat(eigOrth(data.T))
        self._data = data
        if self._conponent == -1:
            self._conponent = data.shape[0]
        W = getRandomW(data.shape[0],self._conponent)
        W = eigOrth(W.T).T
        MAX_T = 10000
        
        for i in range(W.shape[1]):
            w = W[:,i]
            j,t  = 0,1
            while (j < MAX_T) and (t > 1e-8):
                oldw = w.copy()
                w = np.mean(data*GFunction(w.T*data).T,1) - np.mean(gFunction(w.T*data))*w
                temp = np.zeros((W.shape[0],1))
                for k in range(i):
                    temp += float(w.T*W[:,k])*W[:,k]
                w = w - temp
                w = w/math.sqrt(w.T*w)
                W[:,i] = w
                t = distance(w,oldw)
                print(i+1,t)
                j += 1
        self._W = W
        return (self._W.T*data)
    
    def transfer(self,data):
        data = preprocessing.scale(data.T)
        data = np.mat(eigOrth(data.T))
        return (self._W.T*data)
    
    def calculateObj(self):
        data = self._data
        firstPart = np.mean(GFunction(self._W.T.dot(data)),1)
        x = np.arange(-data.shape[1]/2000,data.shape[1]/2000,0.001)
        y = np.mat(np.mean(scipy.stats.norm.pdf(x,0,1)))
        K = np.mean(GFunction(y))
        ICAPart = np.multiply((firstPart - K),(firstPart - K))
        
        diffData = makeDiff(data)
        SlowPart = np.zeros((1,self._W.shape[0]))
        for i in range(self._W.shape[0]):
            w = self._W[:,i]
            secondPart = (w.T*diffData*diffData.T*w)
            SlowPart[0,i] = float(secondPart)
        print("ICA ICAPart:\n",ICAPart)
        print("ICA SlowPart:\n",np.ravel(SlowPart))


if __name__ == '__main__':
#========================================================================
#Load the data and make them the same size
    file1 = "/Users/zhuxiaoxiansheng/Desktop/SICA_data/LDC2017S07.clean.wav" 
    file2 = "/Users/zhuxiaoxiansheng/Desktop/SICA_data/LDC2017S10.embed.wav"
    file3 = "/Users/zhuxiaoxiansheng/Desktop/SICA_data/LDC93S1.wav"

    noise1 = sio.loadmat(u'/Users/zhuxiaoxiansheng/Desktop/SICA_data/noise2.mat')['noise2']  
    time2,noise2 = LoadSound(file2)
    
    time1,data1 = LoadSound(file1)
    time2,data2 = LoadSound(file3)
    data3 = sio.loadmat(u"/Users/zhuxiaoxiansheng/Desktop/SICA_data/voice.mat")['voice']
    
    time1 = time1[1000:-1000]
    data1 = np.mat(data1[1000:-1000])  
    data2 = np.mat(data2[3000:3000+len(time1)])
    data3 = np.mat(data3[0,5000:5000+len(time1)])
    noise2 = np.mat(noise2[0:len(time1)])
 
    data = np.zeros((5,len(time1)))

#=======================================================================
#add the three sounds between each other ,create three mix sounds 
    data1 = preprocessing.scale(data1.T).T
    data2 = preprocessing.scale(data2.T).T
    data3 = preprocessing.scale(data3.T).T
    noise1 = preprocessing.scale(noise1.T).T
    noise2 = preprocessing.scale(noise2.T).T
   
    data[0,:] = data1*10
    data[1,:] = data2*10
    data[2,:] = data3*10
    data[3,:] = noise1*1
    data[4,:] = noise2*5

    A = getRandomW(5,5)
    dataMerage = A.dot(data)

#=======================================================================
    ica = ICA()
    a = ica.fit_transform(dataMerage)


    print("//==========================================================//")
    print("//=====================initial voices======================//")
    ShowRes(data)
    print("//======================mixed voices=======================//")
    ShowRes(dataMerage)
    print("//=========================ICA============================//")
    ShowRes(a)
    ica.calculateObj()

    plotSound(time1,dataS1)
    plotSound(time2,dataS2)
    plotSound(time3,dataS3)     #plot sounds of unmixing by ica
