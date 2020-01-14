import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import time

start = time.perf_counter()

#loading the dataset
dataset = pd.read_csv('mnist_train.csv')

#specifying predictor and response
# predictor having shape as (60000,784)
x = dataset.iloc[:, 1:].values 
  
# reponse having shape as (60000,)
y = dataset.iloc[:, 0].values

#splitting the dataset into training and validation dataset
#taking 75% training data, 25% for test data
#taking 85% training data, 15% for test data
#taking 90% training data, 10% for test data
xtrain, xval, ytrain, yval = train_test_split( 
		x, y, test_size = 0.1, random_state = 0)

#training the algo with 100 trees and 10 jobs parallel
clf = RandomForestClassifier(n_estimators=200,
	bootstrap=True, max_features='sqrt', n_jobs=25)
clf.fit(xtrain,ytrain)

#checking on validation dataset
# RFC accuracy Score on validation dataset : 0.975167
# 0.973 with bootstrap=True, max_features='sqrt'
acc = clf.score(xval,yval)
print('RFC accuracy Score on validation dataset :',acc)

# test data set with 10000 observations
dataset_test = pd.read_csv('mnist_test.csv')
x_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

#checking on test dataset
# RFC accuracy Score on test dataset : 0.9701
# no difference with bootstrap=True, max_features='sqrt'
acc_test = clf.score(x_test,y_test)
print('RFC accuracy Score on test dataset :',acc_test)

print("Time Taken:",round(time.perf_counter(),2)) #126.06s