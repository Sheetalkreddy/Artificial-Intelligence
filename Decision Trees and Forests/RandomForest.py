# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 11:08:05 2018

@author: sheet_000
"""

# -*- coding: utf-8 -*-

""" PLEASE CHECK THE BOTTOM FOR OPTIONS"""


import numpy as np
import random
import math


#print data[0]
K = 1
     
class node :     
    def __init__(self, isClass , index , value) :
        self.isClass = isClass
        
        self.left = None
        self.right = None
        self.index = index
        self.value = value
        self.leaf = True
        self.data = None
        self.depth = 0
        self.groups = None
        
    

def Decision_Tree(train,test,depth,size,K, cho) :
    root = build(train, depth,size, 0 , K, cho)
    results = []
    correct =0.0
    for row in test :
        results.append(predict(root, row))
        #print predict(root,row),row[-1]
        if(predict(root, row)) == row[-1] :
            correct = correct +1.0
            
    return results , 100.0*correct/len(test)
    
    
def build(train, depth , size , d , K, cho) :
    classes , counts = np.unique(train[:,-1] , return_counts = True)
    cls = classes[np.argmax(counts)]
    if (d >= depth ) :
        return  node(cls,None, None)
    elif ( train.shape[0] <  size) :
        return node(cls,None,None)
    elif ( classes.shape[0] ==1 )  :
        
        return node(cls,None,None) 
    else :
         x= get_best(train, K, cho)
         y = node(cls,x[0],x[1])
         
         y.index = x[0]
         y.value = x[1]
         y.data = train
         y.groups = x[2]
         y.depth = d
         y.left = build( y.groups[0] , depth  ,size , d+1, K, cho)
         y.right = build( y.groups[1] , depth  ,size , d+1, K, cho)
         return y
                   
        

def predict(tree , row):
    if(tree.left ==None) and (tree.right ==None) :
        return tree.isClass
    else :
        if( row[tree.index] < tree.value) :
            return predict(tree.left , row)
            
        else :   
            return predict(tree.right , row)
            
    
def split(data, index , value) :
    left = []
    right = []
    for row in data :
        if(row[index] < value) :
            left.append(row)
        else:
            right.append(row)
    return np.array(left),np.array(right)

def get_best(data, K, cho):
    #print "In get Best"
    classes , counts = np.unique(data[:,-1] , return_counts = True)
    inds = []
    vals =[]
    ents = []
    step =10
     
    for index in range(data.shape[1] -1 ):
        values = np.linspace(min(data[:,index]) , max(data[:,index]) , step)
        
        for value in values :
            groups = split(data , index, value)
            grpent = group_ent(groups)
            inds.append(index)
            vals.append(value)
            ents.append(grpent)
            
    num = cho
    m = []
    for i in range(num):
        m.append( i)
        
    choice = random.choice(m)    
    print choice
    ch = np.argsort(ents)[choice]

    
       
    return inds[ch] , vals[ch] , split(data , inds[ch] , vals[ch])      
            
def group_ent(groups) :
    
    grpentropy = 0
    normSize = np.array([ len(groups[0]) , len(groups[1])])
    normSize = (1.0*normSize)/sum(normSize)
    #print len(groups)  , len(groups[0])
    
    
    for i in range(len(groups)) :
        row = np.array(groups[i])
        if (row.shape[0] == 0) :
            probs = []
            entropy = 0
        else :   
            classes ,counts = np.unique(row[:,-1] , return_counts = True)
        
            probs = counts/float(sum(counts))
            entropy = 0
        for p in probs :
            if( p !=0) :
              entropy -= p*math.log(p,2.0)
            else :
                entropy = 0
        grpentropy   += normSize[i] * entropy    
    return grpentropy        

def Bagging(train,test ,depth ,size, K ,cho):
   
    train_set  =        [[] for i in range(K)]
    for i in range(len(train)):
        rem = i%K
        train_set[rem].append(train[i])
        
    result_set = [[] for i in range(K)]
    
    
    for j in range(K) :
              print "Building tree" , j+1
              train_set[j] = np.array(train_set[j])
              result_set[j] = np.array(Decision_Tree(train_set[j], test, depth, size,K,cho)[0])  
              

    result_set = np.array(result_set)          
    final_set = [0 for i in range(len(test))]
    for k in range(len(final_set)):
         classes , counts = np.unique(result_set[:,k] , return_counts = True)
         print classes, counts
         final_set[k] = classes[np.argmax(counts)]
    correct = 0
    for row in test :
         if final_set[i] == row[-1]:
             correct +=1
    return (100.0*correct)/len(test)     
    
  
def splitData(dataset,c) :
    
         train=[]
         test = []
         num = len(dataset)
         if (c==1) :
            random.shuffle(dataset)
            for k in range(len(dataset)) :
                if k%10 == 0:
                    test.append(dataset[k])
                else :
                     train.append(dataset[k])
                     
         elif(c==2) :
             num2 = len(dataset)/11
            
             small = []
             for i in range(11) :
                 p =i*num2
                 for j in range(500)  :
          
                    small.append(dataset[p+j])
             random.shuffle(small)
             num = len(small)
             train = small[0 : int(num*0.9)]
             test = small[int(num*0.9)  :]
            
         return train , test     
      
            
            
if __name__ == "__main__":
         c=1
         filename = 'data_banknote_authentication.csv'
         data = np.genfromtxt(filename, delimiter = ',')
         
  
      
         train , test = splitData(data,c)  
         train = np.array(train)
         test = np.array(test)
   
    
        ##RandomForest   ( change last varuable to choice number)
         print Bagging(train,test,5,100,13,5)
  