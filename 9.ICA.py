import wave
import os
import numpy as np
from sklearn.decomposition import FastICA
import math
import matplotlib.pyplot as plt
from numpy import random
from sklearn import preprocessing 

def LoadSound(path):
    f = wave.open(path,'rb')
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)               #read the wav file
    waveData = np.fromstring(strData,dtype=np.int16) 
    waveData = waveData*1.0/(max(abs(waveData)))  #normalize the sound wave
    time = np.arange(0,nframes*nchannels)*(1.0 / framerate)
    return time.tolist(),waveData.tolist() 

def plotSound(time,data):                         #plot the sound 
    plt.figure()
    plt.plot(time,data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Single channel wavedata")
    plt.grid('on')

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
    length = len(W)
    s = 0.
    for i in range(length):
        s += abs(float(W[i]) - float(oldW[i]))
    return s   

class ICA:                                      #ICA
    def __init__(self,conponent = -1):
        self._W = []
        self._conponent = conponent
        
    def fit_transform(self,data):
        data = preprocessing.scale(data.T)
        data = np.mat(eigOrth(data.T))
        if self._conponent == -1:
            self._conponent = data.shape[0]
        W = getRandomW(data.shape[0],self._conponent)
        W = eigOrth(W)
        MAX_T = 10000
        
        for i in range(W.shape[1]):
            w = W[:,i]
            j,t  = 0,1
            while (j < MAX_T) and (t > 1e-7):
                oldw = w.copy()
                w -= (np.mean(np.multiply(data,GFunction(w.T*data)),1) - np.mean(w.T*data)*w)
                temp = np.zeros((W.shape[0],1))
                for k in range(i):
                    temp += float(w.T*W[:,k])*W[:,k]
                w = w - temp
                w = w/math.sqrt(w.T*w)
                W[:,i] = w
                t = distance(W[:,i],oldw)
                print(i+1,t)
                j += 1
        self._W = W
        return self._W.T*data
    
    def transfer(self,data):
        data = preprocessing.scale(data.T)
        data = np.mat(eigOrth(data.T))
        return self._W.T*data


if __name__ == '__main__':
#========================================================================
#Load the data and make them the same size
    file1 = "/Users/zhuxiaoxiansheng/Desktop/LDC2017S07.clean.wav" 
    file2 = "/Users/zhuxiaoxiansheng/Desktop/LDC2017S10.embed.wav"
    file3 = "/Users/zhuxiaoxiansheng/Desktop/LDC93S1.wav"
    time1,data1 = LoadSound(file1)
    time2,data2 = LoadSound(file2)
    time3,data3 = LoadSound(file3)
  
    time1 = time1[1000:-1000]
    data1 = data1[1000:-1000]
    time2 = time1
    time3 = time1
    data2 = data2[0:len(time1)]
    data3 = data3[3000:3000+len(time1)]

#=======================================================================
#add the three sounds between each other ,create three mix sounds 
    dataMerage1 = np.vstack((np.mat(data1),np.mat(data2)))
    dataMerage1 = (np.ravel(dataMerage1.sum(axis = 0))).tolist()  
    dataMerage2 = np.vstack((np.mat(data3),np.mat(data2)))
    dataMerage2 = (np.ravel(dataMerage2.sum(axis = 0))).tolist()
    dataMerage3 = np.vstack((np.mat(data1),np.mat(data3)))
    dataMerage3 = (np.ravel(dataMerage3.sum(axis = 0))).tolist()
    
    plotSound(time1,data1)
    plotSound(time2,data2)
    plotSound(time3,data3)      #plot three mix sounds

#=======================================================================
    data = np.vstack((np.mat(dataMerage1),np.mat(dataMerage2)))
    data = np.vstack((data,np.mat(dataMerage3)))
  
    sica = SICA()
    dataS = sica.fit_transform(data.T).T

    dataS1 = dataS[0]
    dataS1 = np.ravel(dataS1).tolist()
    dataS2 = dataS[1]
    dataS2 = np.ravel(dataS2).tolist()
    dataS3 = dataS[2]
    dataS3 = np.ravel(dataS3).tolist()

    plotSound(time1,dataS1)
    plotSound(time2,dataS2)
    plotSound(time3,dataS3)     #plot sounds of unmixing by ica
