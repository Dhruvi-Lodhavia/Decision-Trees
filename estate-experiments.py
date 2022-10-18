
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split


np.random.seed(42)
# reading the data from the file
real_estate = pd.read_excel("Real estate valuation data set.xlsx")
real_estate = real_estate.drop(columns=['No'])
# seperating X and y
X = real_estate[real_estate.columns[:-1]]
y = real_estate[real_estate.columns[-1]]

print("***********part(a)-my tree***********")
# using this on my tree
cutoff = int(0.7*len(X))
# dividing the data into 30-70 percent for test and train
X_train = X[:cutoff]
X_test = X[cutoff:].reset_index(drop=True)
y_train = y[:cutoff]
y_test = y[cutoff:].reset_index(drop=True)

    
    
# fitting, predicting and printing the metreics
for criteria in ['information_gain']:
    tree = DecisionTree(criterion=criteria) #Split based on Inf. Gain
    tree.fit(X_train, y_train)
    y_hat = tree.predict(X_test)
    tree.plot()
    print('Criteria :', criteria)
    print('RMSE: ', rmse(y_hat, y_test))
    print('MAE: ', mae(y_hat, y_test))


# print("partb")
# Part(b) sklearn tree
print('****************Part(b) sklearn tree***********')
# dividing test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# using sklearn decision tree regressor
sktree = DecisionTreeRegressor()
# fitting it
sktree = sktree.fit(X_train,y_train)

y_hat = sktree.predict(X_test) 

print('RMSE: ', rmse(y_hat, y_test))
print('MAE: ', mae(y_hat, y_test))

print("\nDecisionTree results\n")
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot 
export_graphviz(sktree, out_file = "tree.dot")