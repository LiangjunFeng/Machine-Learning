#!/usr/bin/env python3
# -*-coding: utf-8-*-
# Author : LiangjunFeng
# Blog   : http://my.csdn.net/Liangjun_Feng
# GitHub : https://www.github.com/LiangjunFeng
# File   : linear model.py
# Date   : 2017/08/31 3:12
# Version: 0.1
# Description: decision tree 

import math
import copy

# Tree Node class
class TNode:
    def __init__(self,D = None,A = None,Flag = None,Next = None,Label = None,Pre = None):
        self._D = D
        self._Flag = Flag
        self._Label = Label
        self._Next = []
        if Next != None:
            self._Next.append(Next)
        self._A = A
        self._Pre = Pre
        
    def get_Pre(self):
        return self._Pre
        
    def get_A(self):
        return self._A
    
    def get_D(self):
        return self._D
    
    def get_Flag(self):
        return self._Flag
    
    def get_Label(self):
        return self._Label
    
    def get_Nlength(self):
        return len(self._Next)
    
    def get_Next(self,i):
        return self._Next[i]
    
    def set_Pre(self,Pre):
        self._Pre = Pre
    
    def set_A(self,A):
        self._A = A
        
    def set_D(self,D):
        self._D = D
    
    def set_Flag(self,Flag):
        self._Flag = Flag
        
    def set_Label(self,Label):
        self._Label = Label
    
    def set_Next(self,Next):
        self._Next.append(Next)
    
    def printall(self,i):
        print('A     ',self._A ,i,self)
        print('Data  ',self._D,len(self._D))
        print('Flag  ',self._Flag)
        print('Label ',self._Label)
        print('Next  ',self._Next)
        print('Pre   ',self._Pre)
        print()
        

#decision Tree class   
class decision_Tree:
    def __init__(self,D,A):
        self._root = TNode(D,A)
        self.D_Preprocess()
        decision_Tree.tree_Grow(self._root)         
    
    def fit(self,D):
        li,i,Data = [[]],0,copy.deepcopy(D)
        while i < len(Data):
            head = self._root
            while head.get_Flag() != 'leaf':
                j = decision_Tree.find_label(head,head.get_Label())
                head = head.get_Next(Data[i][0][j])
                del(Data[i][0][j])
                j += 1
            while int(head.get_Label()) > (len(li)-1):
                li.append([])
            li[int(head.get_Label())].append(D[i])
            i += 1
        return li                       
        
    def D_Preprocess(self):
        i,Data,length = 0,self._root.get_D(),len(self._root.get_D())
        while i < length:
            j = 0
            while j < len(Data[i][0]):
                if Data[i][0][j]%1 > 0:
                    break
                j += 1
            if j != len(Data[i][0]):
                j = 0
                threshold = (max(Data[i][0]) + min(Data[i][0]))/2
                while j < len(Data[i][0]):
                    if Data[i][0][j] < threshold:
                        Data[i][0][j] = 0
                    else:
                        Data[i][0][j] = 1
                    j += 1
            i += 1
        self._root.set_D(Data)
    
    @classmethod 
    def find_label(cls,Node,t):
        i = 0
        A = Node.get_A()
        while i < len(A):
            if A[i] == t:
                return i
            i += 1
        return None                        

    @classmethod               
    def Ent(cls,D):
        res,i,length,Data,count = 0,0,len(D),D,0
        while i < length:
            if Data[i][1] > count:
                count = Data[i][1]
            i += 1
        count = [0]*(count+1)
        i = 0
        while i < length:
            count[Data[i][1]] += 1
            i += 1
        i = 0
        while i < len(count):
            count[i] /= length
            i += 1
        i = 0
        while i < len(count):
            if count[i] == 0:
                i += 1
            if i < len(count):
                res += count[i] * math.log(count[i],2)
            i += 1
        return -1*res

    @classmethod
    def a_Ent(cls,D,a):
        i,count,Data,length,label = 0,0,D,len(D),a
        while i < length:
            if Data[i][0][label] > count:
                count = Data[i][0][label]
            i += 1
        count = [0]*(count+1)
        i = 0
        while i < length:
            count[Data[i][0][label]] += 1
            i += 1
        i = 0
        while i < len(count):
            count[i] /= length
            i += 1
        i = 0
        Dv,DV = [[]]*len(count),0
        while i < len(count):
            j = 0 
            Dv[i] = []
            while j < length:
                if Data[j][0][label] == i:
                    Dv[i].append(Data[j])
                j += 1
            if Dv[i] == []:
                i += 1
                continue
            DV += count[i]*decision_Tree.Ent(Dv[i])
            i += 1
        return DV
    
    @classmethod
    def Gain(cls,D,a):
        return decision_Tree.Ent(D)-decision_Tree.a_Ent(D,a)
    
    @classmethod
    def most_Class(cls,D):
        i,count = 0,0
        while i < len(D):
            if D[i][1] > count:
                count = D[i][1]
            i += 1
        i,count = 0,[0]*(count + 1)
        while i < len(D):
            count[D[i][1]] += 1
            i += 1
        i,j,t = 0,0,count[0]
        while i < len(count):
            if count[i] > t:
                t = count[i]
                j = i
            i += 1
        return j
    
    @classmethod
    def delete_item(cls,Node,t):
        i,Data,A = 0,copy.deepcopy(Node.get_D()),copy.deepcopy(Node.get_A())
        while i < len(Data):
            del(Data[i][0][t])
            i += 1
        del(A[t])
        Node.set_D(Data)
        Node.set_A(A)
        return Node
        
    @classmethod
    def branches(cls,Node,t):
        i,count,Data,A = 0,0,copy.deepcopy(Node.get_D()),copy.deepcopy(Node.get_A())
        
        while i < len(Data):
            if Data[i][0][t] > count:
                count = Data[i][0][t]
            i += 1
        i,NData,li = 0,[[]]*(count+1),[]
        while i < (count+1):
            j,NData[i] = 0,[]
            while j < len(Data):
                if Data[j][0][t] == i:
                    NData[i].append(Data[j])
                j += 1
            a = TNode(NData[i],A,Flag= None,Label = None,Next = None,Pre = Node)
            a = decision_Tree.delete_item(a,t)
            Node.set_Next(a)
            li.append(a)
            i += 1
        return li   
    
    @classmethod
    def tree_Grow(cls,Node):
        Data = copy.deepcopy(Node.get_D())
        if Data == []:
            Node.set_Flag('leaf')
            Data = Node.get_Pre().get_D()
            Node.set_Label(decision_Tree.most_Class(Data)) 
            return 
        
        i = 0
        while i+1 < len(Data):
            if Data[i][0] == Data[i+1][0]:
                i += 1
            else:
                break                      
        if Node.get_A() == [] or (i+1) == len(Data):
            Node.set_Flag('leaf')
            Node.set_Label(decision_Tree.most_Class(Data))
            return
        
        i = 0
        while i+1 < len(Data):
            if Data[i][1] == Data[i+1][1]:
                i += 1
            else:
                break
        if i+1 == len(Data):
            Node.set_Flag('leaf')
            Node.set_Label(Data[0][1])
            return
        
        i,j,t = 0,0,0
        while i < len(Data[0][0]):
            a = decision_Tree.Gain(Data,i)
            if a > j:
                j = a
                t = i
            i += 1
        Node.set_Flag('Node')
        Node.set_Label(Node.get_A()[t])   
                
        li = decision_Tree.branches(Node,t)
        i = 0
        while i < len(li):
            decision_Tree.tree_Grow(li[i])
            i += 1
    
    @staticmethod
    def printall(Node,i):
        if Node is None:
            return
        else:
            if Node.get_Flag() != 'N':
                Node.printall(i)
            i,num = 0,Node.get_Nlength()
            while i < num:
                decision_Tree.printall(Node.get_Next(i),i)
                i += 1
            
        
                   
        
'''
D = [[[0,0,0,0,0,0],1,1],[[1,0,1,0,0,0],1,2],[[1,0,0,0,0,0],1,3], 
     [[0,1,0,0,1,1],1,6],[[1,1,0,1,1,1],1,7],[[0,2,2,0,2,1],0,10],
     [[2,1,1,1,0,0],0,14],[[1,1,0,0,1,1],0,15],[[2,0,0,2,2,0],0,16],[[0,0,1,1,1,0],0,17]]

d = [[[0,0,1,0,0,0],1,4],[[2,0,0,0,0,0],1,5],[[1,1,0,0,1,0],1,8],[[1,1,1,1,1,0],0,9],
     [[2,2,2,2,2,0],0,11],[[2,0,0,2,2,1],0,12], [[0,1,0,1,0,0],0,13]]

A = ['色泽','根蒂','敲声','纹理','脐部','触感']
T = decision_Tree(D,A)    
print(T.fit(d))   
'''           
        

            
                
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    