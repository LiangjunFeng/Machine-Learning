#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : support_vector_machine.py
# Date   : 2017/09/21 11:12
# Version: 0.1
# Description: support_vector_machine

import numpy
import cvxopt
import matplotlib.pyplot as plt

def linearKernel(x1,x2):        #the linear kernel function
    return numpy.dot(x1,x2)

def gaussianKernel(x1,x2,sigma = 3):       #the guassian kernel function
    return numpy.exp(-numpy.linalg.norm(x1-x2)**2/(2*(sigma ** 2)))

def makeKernelMatrix(input_,kernel,par = 3):      #set up kernel matrix according to the kernel
    num = input_.shape[0]
    K = numpy.zeros((num,num))
    for i in range(num):
        for j in range(num):
            if kernel == 'linearKernel':
                K[i,j] = linearKernel(input_[i],input_[j])
            else:
                K[i,j] = gaussianKernel(input_[i],input_[j],par)
    return K

def calculateA(input_,label,C,K):          #using cvxopt bag figure out the convex quadratic programming
    num = input_.shape[0]
    P = cvxopt.matrix(numpy.outer(label,label)*K)
    q = cvxopt.matrix(numpy.ones(num)*-1)
    A = cvxopt.matrix(label,(1,num))
    b = cvxopt.matrix(0.0)
    
    if C is None:
        G = cvxopt.matrix(numpy.diag(numpy.ones(num)*-1))
        h = cvxopt.matrix(numpy.zeros(num))
    else:
        temp1 = numpy.diag(numpy.ones(num)*-1)
        temp2 = bp.identity(num)
        G = cvxopt.matrix(numpy.vstack(temp1,temp2))
        temp1 = numpy.zeros(num)
        temp2 = numpy.ones(num)*self.C
        h = cvxopt.matrix(numpy.hstack(temp1,temp2))        #P\q\A\b\G\h are parameters of cvxopt.solvers.qp() function

    solution = cvxopt.solvers.qp(P,q,G,h,A,b)               #figure out the model
    
    a = numpy.ravel(solution['x'])      #transfer the 'a' into a vector
    return a

def calculateB(a,supportVectorLabel,supportVector,K,indexis):    #calculate the parameter 'b'
    b = 0
    for i in range(len(a)):
        b += supportVectorLabel[i]
        b -= numpy.sum(a*supportVectorLabel*K[indexis[i],supportVector])
    b /= len(a)                         #set the  'b' with the mean value  
    return b

def calculateWeight(kernel,features,a,supportVector,supportVectorLabel):     #calculate the model's weights according to a
    if kernel == linearKernel:
        w = numpy.zeros(features)
        for i in range(len(a)):
            w += a[i]*supportVectorLabel[i]*supportVector[i]     #linear kernel calculate as the linear model
    else:
        w = None     #nonlinear nodel need not calculate ,we will process it later
    return w

class SVM:           #SVM class
    def __init__(self,kernel = linearKernel,C = None):    #init
        self.kernel = kernel            #the kernel's algrithm
        self.C = C                      #the soft margin's parameter
        self.a = None                   #the process parameter
        self.b = 0                      #the model's risdual
        self.w = []                     #the model's weights
        self.supportVector = []         #the support vector
        self.supportVectorLabel = []    #the support's label
        if self.C is not None:
            self.C = float(self.C)
            
    def fit(self,input_,label):         #fit data
        samples,features = input_.shape
        
        K = makeKernelMatrix(input_,self.kernel)          #calculate the kernel matrix
        a = calculateA(input_,label,self.C,K)             #calculate the process parameter 'a'
        supportVector = a > 1e-5                          #if a < 1e-5,then think a is 1
        indexis = numpy.arange(len(a))[supportVector]     #the support vectors' indexis
        self.a = a[supportVector]                         #the support vectors' a
        self.supportVector = input_[supportVector]        #the support vectors
        self.supportVectorLabel = label[supportVector]    #the support vectorss' label
        
        print(len(self.a),' support vectors out of ',samples,' points')
        
        self.b = calculateB(self.a,self.supportVectorLabel,supportVector,K,indexis)   #calculate the model's risdual
        self.w = calculateWeight(self.kernel,features,self.a,self.supportVector,self.supportVectorLabel)    #calculate the model's weights
        
    def predict(self,input_):            #predict function
        if self.w is not None:
            return numpy.dot(inpt_,self.w) + self.b
        else:
            predictLabel = numpy.zeros(len(input_))
            for i in range(len(input_)):
                s  = 0
                for a,sv_y,sv in zip(self.a,self.supportVectorLabel,self.supportVector):
                    s += a * sv_y * self.kernel(input_[i],sv)
                predictLabel[i] = s
            return numpy.sign(predictLabel+self.b)
        
def gen_non_lin_separable_data():        #creat the data
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.0,0.8], [0.8, 1.0]]
    X1 = np.random.multivariate_normal(mean1, cov, 60)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 60)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 60)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 60)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2 
     
def split_train(X1, y1, X2, y2):        #split the train data from the original data
     X1_train = X1[:90] 
     y1_train = y1[:90]
     X2_train = X2[:90]
     y2_train = y2[:90]
     X_train = np.vstack((X1_train, X2_train))
     y_train = np.hstack((y1_train, y2_train))
     return X_train, y_train
        
def split_test(X1, y1, X2, y2):         #split the test data from the original data
     X1_test = X1[90:]
     y1_test = y1[90:]
     X2_test = X2[90:]
     y2_test = y2[90:]
     X_test = np.vstack((X1_test, X2_test))
     y_test = np.hstack((y1_test, y2_test))
     return X_test, y_test     
        
def test_non_linear():                  #test
     X1, y1, X2, y2 = gen_non_lin_separable_data()
     X_train, y_train = split_train(X1, y1, X2, y2)
     print(y_train,'$$$')
     X_test, y_test = split_test(X1, y1, X2, y2)
     clf = SVM(gaussianKernel)

     X_train = numpy.array(X_train)
     y_train = numpy.array(y_train)
     print(y_train,'$$$')
     clf.fit(X_train, y_train)
     y_predict = clf.predict(X_test)
     plt.title('scatter diagram')
     for i in range(len(X_test)):
         if y_predict[i] == 1:
             plt.plot(X_test[i,0],X_test[i,1],'ro')
         else:
             plt.plot(X_test[i,0],X_test[i,1],'go')
     plt.show()



if __name__ == "__main__":
    test_non_linear()      
        
