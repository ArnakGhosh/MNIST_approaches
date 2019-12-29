import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
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

# test data set with 10000 observations
dataset_test = pd.read_csv('mnist_test.csv')
x_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

# building the model using polynomial kernel, as in the case
# of image classification data
# clf = svm.SVC(gamma=0.1, kernel='poly') score = 0.98044
# clf = svm.SVC(kernel='poly') score = 0.97767 having default gamma as 'scale'
clf = svm.SVC(gamma='auto', kernel='poly') #score = 0.981 with 10% test-train split
clf.fit(xtrain,ytrain)

# Calculating Accuracy of trained Classifier
acc = clf.score(xval,yval)
print("Accuracy in predicting with validation data set :",acc)

#predicting with test data
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print("Accuracy in predicting with test data set :",accuracy) #accuracy score = 0.9782

print("Time Taken:",round(time.perf_counter(),2)) # time taken = 612.29s