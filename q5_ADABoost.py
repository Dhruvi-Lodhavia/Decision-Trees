"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeClassifier

# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))

y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")
y = y.replace(to_replace =[0,1],value=[-1,1]).astype('category')

# using sklearn decision tree classifier
tree = DecisionTreeClassifier
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
# print(y_hat)
# [fig1, fig2] = Classifier_AB.plot()
print('Criteria :', "information gain")
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features

print("part(b)") 
print("****************on iris dataset*****************")
iris = pd.read_csv("Iris.csv", header = None, names = ["sepal length", "sepal width", "petal length", "petal width", "label"])
X1 = iris.iloc[:,1]
X2 = iris.iloc[:,3]
X = pd.concat([X1,X2],axis = 1)
# replacing virginica as -1 and setosa and versicolor as 1 label
y = iris[iris.columns[-1]].astype('category')
y = y.replace(to_replace =["Iris-setosa","Iris-virginica","Iris-versicolor"],value=[1,-1,1]).astype('category')


XY = pd.concat([X, y.rename("y")], axis=1)
XY = XY.reset_index(drop=True)
# shuflling the data set with seed 42
np.random.seed(42)
nums = np.arange(len(XY))
np.random.shuffle(nums)
XY = XY.iloc[nums].reset_index(drop=True)
# dividing dataset into test and train in ration 40-60
cutoff = int(0.6*len(X))
X = XY[XY.columns[:-1]]
y = XY.iloc[:,-1]
X_train = X[:cutoff]
X_test = X[cutoff:].reset_index(drop=True)
y_train = y[:cutoff]
y_test = y[cutoff:].reset_index(drop=True)

# for 3 n_estimators
n_estimators = 3
criteria = 'entropy'

# tree with depth 1
print("***********for decision stump**************")
for criteria in ['information_gain']:
    tree1 = DecisionTreeClassifier(criterion = 'entropy',max_depth=1) #Split based on Inf. Gain
    tree1.fit(X_train, y_train)
    y_hat = tree1.predict(X_test)
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y_test))
    for cls in y.unique():
        print('Precision: ', precision(y_hat, y_test, cls))
        print('Recall: ', recall(y_hat, y_test, cls))
print("*******************for adaboost*****************")
# adaboost classifier
tree = DecisionTreeClassifier
#tree = DecisionTree(criterion=criteria,max_depth = 1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))