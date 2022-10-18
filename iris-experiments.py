import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Read IRIS data set
# ...
# 
# reading the dataset from iris.csv
iris = pd.read_csv("Iris.csv", header = None, names = ["sepal length", "sepal width", "petal length", "petal width", "label"])
X = iris[iris.columns[:-1]]
y = iris[iris.columns[-1]].astype('category')

# part a
print("***********part(a)*************")
# dividing test and train in 30-70 percent
cutoff = int(0.7*len(X))

X_train = X[:cutoff]
X_test = X[cutoff:].reset_index(drop=True)
y_train = y[:cutoff]
y_test = y[cutoff:].reset_index(drop=True)

# running the iris data set on my tree
for criteria in ['information_gain']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))
        
# 5 fold cross validation
print("****************part(b) - 5 fold cross validation*************")
accuracy_cv = []  
# function defined that returns data in parts(4 and 1) depending on k value given
def fivefoldcvdataset(XY,k):
    l = int(len(XY)/5)
    
    test = XY[l*k:l*(k+1)]
    test = test.reset_index(drop=True)
    # if test part is first
    if(k==0):
        train = XY[l:]
    else:
        train_p1 = XY[0:l*k]
        train_p2 = XY[l*(k+1):]
        train = pd.concat([train_p1,train_p2],axis=0)
    # returning teh two parts
    train = train.reset_index(drop=True)
    return train,test

for k in range(5):
    # dividing test and train into 5 different datasets
    train,test = fivefoldcvdataset(iris,k)
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]].astype('category')
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]].astype('category')
    # calling decision tree
    tree = DecisionTree(criterion='information_gain',max_depth = 8)
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    # returning accuracy found and appending in 
    accuracy_cv.append(accuracy(y_hat, y_test))
    
# returing max accuracy
m_acc = max(accuracy_cv)
print("accuracy is ",m_acc)
    
    

print("******************nested cv***************")
iris = pd.read_csv("Iris.csv", header = None, names = ["sepal length", "sepal width", "petal length", "petal width", "label"])
best_depth = 0
dep = []
accuracy_list = []
max_acc = 0
# running nested cv for 5 loops
for fold in range(5):
    # dividing dataset into test and trai n
    train,test = fivefoldcvdataset(iris,fold)
    X_train = train[train.columns[:-1]]
    y_train = train[train.columns[-1]].astype('category')
    X_test = test[test.columns[:-1]]
    y_test = test[test.columns[-1]].astype('category')
    XY = pd.concat([X_train,y_train],axis = 1)   
    for depth in range(5):
        #varying depth
        acc = []
        for k in range(3):
            # varying folds for validation set and dividing train set in test and validate and fitting it in the tree
            train1,valid = fivefoldcvdataset(XY,k)
            X_train1 = train1[train1.columns[:-1]]
            y_train1 = train1[train1.columns[-1]].astype('category')
            X_valid = valid[valid.columns[:-1]]
            y_valid = valid[valid.columns[-1]].astype('category')
            tree = DecisionTree(criterion='information_gain',max_depth = depth+1)
            tree.fit(X_train1, y_train1)
            y_hat = tree.predict(X_valid)
            # appending accuracy for each iteration
            acc.append(accuracy(y_hat, y_valid))
        # calc avg for each k
        avg_acc = sum(acc)/len(acc)
        # finding max acc for each depth and its values
        if(max_acc < avg_acc):
            max_acc = avg_acc
            best_depth = depth+1
            mytree = tree
            # mytree.plot()
            # myXtrain = X_train1
            # myytrain = y_train1
            # myXtest = X_test
            # myytest = y_test
        dep.append(best_depth)
        accuracy_list.append(max_acc)
for i in range(3):
    print("accuracy is",accuracy_list[i],"for depth",dep[i])




            