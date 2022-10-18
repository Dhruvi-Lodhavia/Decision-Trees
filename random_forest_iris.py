import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here
# loading the iris dataset
iris = pd.read_csv("Iris.csv", header = None, names = ["sepal length", "sepal width", "petal length", "petal width", "label"])
X1 = iris.iloc[:,1]
X2 = iris.iloc[:,3]
# taking columns sepal width and petal width
X = pd.concat([X1,X2],axis = 1)
# converting y type to category
y = iris[iris.columns[-1]].astype('category')

# seperating XY and shuffling it at seed 42
XY = pd.concat([X, y.rename("y")], axis=1)
XY = XY.reset_index(drop=True)
np.random.seed(42)
nums = np.arange(len(XY))
np.random.shuffle(nums)
XY = XY.iloc[nums].reset_index(drop=True)
# dividing it in ration 40-60 for test and train
cutoff = int(0.6*len(X))
X = XY[XY.columns[:-1]]
y = XY.iloc[:,-1]
X_train = X[:cutoff]
X_test = X[cutoff:].reset_index(drop=True)
y_train = y[:cutoff]
y_test = y[cutoff:].reset_index(drop=True)


criteria = 'information_gain'
Classifier_RF = RandomForestClassifier(10, criterion = criteria)
Classifier_RF.fit(X_train, y_train)
y_hat = Classifier_RF.predict(X_test)
Classifier_RF.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))