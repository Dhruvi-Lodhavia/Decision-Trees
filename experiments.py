
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs S
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""


np.random.seed(42)
# creating fake data 
# typeX = True - Real else discrete
# same for typeY
def createfakedata(N,M,typeX,typey):
    if(typeX):
        X = pd.DataFrame(np.random.randn(N, M))
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(M)})
    if(typey):
        y = pd.Series(np.random.randn(N))
    else:
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    return X,y
    
# function for calculating time
# if calc = True then fit
# else then calculating time for predict
def timer(calc,N,M,typeX,typey):
    if(calc):
        #calling the tree and using the fake data to find the difference between starta nd end time in fitting
        tree = DecisionTree(criterion='information_gain') #Split based on Inf. Gain
        X,y = createfakedata(N,M,typeX,typey)
        stim = time.time()
        tree.fit(X, y)
        etim = time.time()
        
    else:
        #calling the tree and using the fake data to find the difference between start and end time in predicting
        tree = DecisionTree(criterion='information_gain') #Split based on Inf. Gain
        X,y = createfakedata(N,M,typeX,typey)
        tree.fit(X, y)
        stim = time.time()
        y_hat = tree.predict(X)
        etim = time.time()
    total = etim - stim
    return total
        
N = [10,20,30,40,50]
M = 5
RR_fit = []
RD_fit = []
DD_fit = []
DR_fit = []
for n in N:
    #RR case - Xtype,ytype = True
    x = timer(True,n,M,True,True)
    RR_fit.append(x)
    RD_fit.append(timer(True,n,M,True,False))
    DD_fit.append(timer(True,n,M,False,False))
    DR_fit.append(timer(True,n,M,False,True))

plt.figure()
plt.plot(N,RR_fit,'-ok',label = 'RR_fit')
plt.plot(N,RD_fit,'-o',label = 'RD_fit')
plt.plot(N,DD_fit,label = 'DD_fit')
plt.plot(N,DR_fit,label = 'DR_fit')
plt.legend()
plt.xlabel('N')
plt.ylabel('Time')
plt.savefig("Fit time vsN(M=5)")

N = 30
M = [2,3,4,5,6,7,8,9,10]
RR_fit = []
RD_fit = []
DD_fit = []
DR_fit = []
for m in M:
    #RR case - Xtype,ytype = True
    RR_fit.append(timer(True,N,m,True,True))
    RD_fit.append(timer(True,N,m,True,False))
    DD_fit.append(timer(True,N,m,False,False))
    DR_fit.append(timer(True,N,m,False,True))
    
plt.figure()
plt.plot(M,RR_fit,'-ok',label = 'RR_fit')
plt.plot(M,RD_fit,'-o',label = 'RD_fit')
plt.plot(M,DD_fit,label = 'DD_fit')
plt.plot(M,DR_fit,label = 'DR_fit')
plt.legend()
plt.xlabel('M')
plt.ylabel('Time')
plt.savefig("Fit time vsM (N=30)")

#predicting
        
N = [10,20,30,40,50]
M = 5
RR_predict = []
RD_predict = []
DD_predict = []
DR_predict = []
for n in N:
    #RR case - Xtype,ytype = True
    x = timer(False,n,M,True,True)
    RR_predict.append(x)
    RD_predict.append(timer(False,n,M,True,False))
    DD_predict.append(timer(False,n,M,False,False))
    DR_predict.append(timer(False,n,M,False,True))

plt.figure()
plt.plot(N,RR_predict,'-ok',label = 'RR_predict')
plt.plot(N,RD_predict,'-o',label = 'RD_predict')
plt.plot(N,DD_predict,label = 'DD_predict')
plt.plot(N,DR_predict,label = 'DR_predict')
plt.legend()
plt.xlabel('N')
plt.ylabel('Time')
plt.savefig("Predict time vsN(M=5)")

N = 30
M = [2,3,4,5,6,7,8,9,10]
RR_predict = []
RD_predict = []
DD_predict = []
DR_predict = []
for m in M:
    #RR case - Xtype,ytype = True
    x = timer(False,N,m,True,True)
    RR_predict.append(x)
    RD_predict.append(timer(False,N,m,True,False))
    DD_predict.append(timer(False,N,m,False,False))
    DR_predict.append(timer(False,N,m,False,True))
    
plt.figure()
plt.plot(M,RR_predict,'-ok',label = 'RR_predict')
plt.plot(M,RD_predict,'-o',label = 'RD_predict')
plt.plot(M,DD_predict,label = 'DD_predict')
plt.plot(M,DR_predict,label = 'DR_predict')
plt.legend()
plt.xlabel('M')
plt.ylabel('Time')
plt.savefig("Predict time vsM (N=30)")


