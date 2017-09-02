#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : linear model.py
# Date   : 2017/08/31 1:57
# Version: 0.1
# Description: code of Three different linear models


'''
1.using least square method figure out linear regression
'''
----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x,y):     
    def preprocess(x):
        X = np.mat(x)
        b = np.mat([1]*len(x))
        X = np.hstack((X,b.T))
        return X

    def cal_w(x,y):
        X = preprocess(x)
        Y = np.mat(y).T
        return (X.T*X).I*X.T*Y     
    
    return preprocess(x)*cal_w(x,y),cal_w(x,y).tolist()


#visiable and output test
x = [[0.697,0.460],[0.774,0.376],[0.634,0.264],[0.608,0.318],[0.556,0.215],[0.403,0.237],[0.481,0.149],
     [0.437,0.211],[0.666,0.091],[0.243,0.267],[0.245,0.057],[0.343,0.099],[0.639,0.161],[0.657,0.198],
     [0.360,0.370],[0.593,0.042],[0.719,0.103]]
y = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]
yy,w = linear_regression(x,y)
i,x1,x2 = 0,[],[]
while i < len(x):
    x1.append(x[i][0])
    x2.append(x[i][1])
    i += 1

i = 0
plt.figure(1)
plt.subplot(121)
plt.title('linear regression')
while i < len(x1):
    if y[i] == 0:
        plt.scatter(x1[i],x2[i],color = 'r')
    elif y[i] == 1:
        plt.scatter(x1[i],x2[i],color = 'g')
    i += 1
a = -(w[2][0]/w[1][0])
b = -(w[0][0]+w[2][0])/w[1][0]
plt.plot([0,1],[a,b])




'''
2.Newton method figure out logistic regression
'''
----------------------------------------------------------------------
def logistic_regression(x,y,error,n):
    def preprocess(x,y):
        X = np.mat(x)
        b = np.mat([1]*len(x))
        X = np.hstack((X,b.T))
        w = [1]*(len(x[0])+1)
        W = np.mat(w).T
        Y = y
        return X,W,Y   
    
    def func_p(X,W):
        a = (X*W).tolist()
        b = float(a[0][0])
        temp = np.exp(b)/(1+np.exp(b))
        return temp
    
    def dfunc(X,Y,W):
        i,num,sum1 = 0,len(X),0
        while i < num:
            temp = Y[i] - func_p(X[i],W)
            sum1 += X[i]*temp
            i += 1
        return sum1*(-1)
    
    def d2func(X,Y,W):
        i,num,sum1 = 0,len(X),0
        while i < num:         
            temp = func_p(X[i],W)*(1 - func_p(X[i],W))
            sum1 += X[i]*(X[i].T)*temp
            i += 1
        sum1 = sum1.tolist()
        return float(sum1[0][0])

    def Newton(x,y,error,n):
        X,W,Y = preprocess(x,y)
        i = 1
        while i < n:
            d1 = dfunc(X,Y,W)
            a = (d1*d1.T).tolist()
            a = float(a[0][0])
            if a < error:
                return W
                break
            temp = dfunc(X,Y,W)
            W = W - temp.T*(d2func(X,Y,W)**(-1))
            i += 1
        if i == n:
            return 'error'
    
    w = Newton(x,y,error,n)
    X,W,Y = preprocess(x,y)
    yy = (X*w).tolist()
    w = w.tolist()
    return w,yy


#visiable and output test           
w,yy = logistic_regression(x,y,0.0001,1000)

i,x1,x2,z = 0,[],[],[]
while i < len(x):
    x1.append(x[i][0])
    x2.append(x[i][1])
    z.append(yy[i][0])
    i += 1

i = 0

plt.subplot(122)
plt.title('logistic regression')
while i < len(x1):
    if y[i] == 0:
        plt.scatter(x1[i],x2[i],color = 'r')
    elif y[i] == 1:
        plt.scatter(x1[i],x2[i],color = 'g')
    i += 1
a = -(w[2][0]/w[1][0])
b = -(w[0][0]+w[2][0])/w[1][0]
plt.plot([0,1],[a,b])



'''
3.Linear Discriminant Analysis for binary classification problem
'''
----------------------------------------------------------------------
def LDA(x,y):
    def preprocess(x,y):
        i = 0
        X0,X1 = [],[]
        while i < len(y):
            if y[i] == 0:
                X0.append(x[i])
            elif y[i] == 1:
                X1.append(x[i])
            i += 1
        return X0,X1
    
    def average(X):
        X = np.mat(X)
        i = 1
        while i < len(X):
            X[0] = X[0] + X[i]
            i += 1
        res = X[0]/i
        return res
    
    def Sw(X0,X1,u0,u1):
        X_0 = np.mat(X0)
        X_1 = np.mat(X1)
        Sw0,i = 0,0
        temp0 = (X_0 - u0)*((X_0 - u0).T)
        while i < len(temp0):
            Sw0 += float(temp0[i,i])
            i += 1
        Sw1,i = 0,0
        temp1 = (X_1 - u1)*((X_1 - u1).T)
        while i < len(temp1):
            Sw1 += float(temp1[i,i])
            i += 1
        return Sw0+Sw1
        
    
    
    X0,X1 = preprocess(x,y)
    u0,u1 = average(X0),average(X1)
    SW = Sw(X0,X1,u0,u1)
    return (SW**(-1)*(u0-u1)).tolist()[0]    

#visiable and output test           
W = LDA(x,y)

i,x1,x2,z = 0,[],[],[]
while i < len(x):
    x1.append(x[i][0])
    x2.append(x[i][1])
    i += 1
i = 0
plt.figure(2)
plt.subplot(121)
plt.title('LDA')
while i < len(x1):
    if y[i] == 0:
        plt.scatter(x1[i],x2[i],color = 'r')
    elif y[i] == 1:
        plt.scatter(x1[i],x2[i],color = 'g')
    i += 1
print(W)

plt.plot([0,-3*W[0]],[0,-3*W[1]])















