import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy import optimize as opt
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
xtrain, xval, ytrain, yval = train_test_split( 
		x, y, test_size = 0.15, random_state = 0)

#shape of the data
#adding constant term in x
# m = len(y)
# ones = np.ones((m,1))
# x = np.hstack((ones, x)) #add the constant
# (m,n) = x.shape

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k

# kVals = range(1, 30, 2)
# accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier

# for k in range(1, 30, 2):
#     # train the k-Nearest Neighbor classifier with the current value of `k`
#     model = KNeighborsClassifier(n_neighbors=k)
#     model.fit(xtrain, ytrain)
#     # evaluate the model and update the accuracies list
#     score = model.score(xval, yval)
#     print("k=%d, accuracy=%.2f%%" % (k, score * 100))
#     accuracies.append(score)
          
# # find the value of k that has the largest accuracy

# i = np.argmax(accuracies)
# print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
# accuracies[i] * 100))

# output from above code

# k=1, accuracy=97.43%
# k=3, accuracy=97.51%
# k=5, accuracy=97.19%
# k=7, accuracy=97.17%
# k=9, accuracy=96.94%
# k=11, accuracy=96.86%
# k=13, accuracy=96.72%
# k=15, accuracy=96.66%
# k=17, accuracy=96.52%
# k=19, accuracy=96.39%
# k=21, accuracy=96.26%
# k=23, accuracy=96.16%
# k=25, accuracy=96.09%
# k=27, accuracy=96.08%
# k=29, accuracy=95.98%
# k=3 achieved highest accuracy of 97.51% on validation data
# Time Taken: 15401.02
# [Finished in 15401.7s]

#taking k=3

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)

dataset_test = pd.read_csv('mnist_test.csv')
x_test = dataset_test.iloc[:, 1:].values
y_test = dataset_test.iloc[:, 0].values

predictions = model.predict(x_test)


# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits

print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))

print ("Confusion matrix")
print(confusion_matrix(y_test,predictions))


print("Time Taken:",round(time.perf_counter(),2))