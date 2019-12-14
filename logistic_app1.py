import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
xtrain, xtest, ytrain, ytest = train_test_split( 
		x, y, test_size = 0.15, random_state = 0)

#shape of the data
#adding constant term in x
m = len(y)
ones = np.ones((m,1))
x = np.hstack((ones, x)) #add the constant
(m,n) = x.shape

#vecorized implementaton of sigmoid function
def sigmoid(z):
  return 1/(1+np.exp(-z))

#cost function
def costFunctionReg(theta, X, y, lmbda):
  m = len(y)
  temp1 = np.multiply(y, np.log(sigmoid(np.dot(X, theta))))
  temp2 = np.multiply(1-y, np.log(1-sigmoid(np.dot(X, theta))))
  return np.sum(temp1 + temp2) / (-m) + np.sum(theta[1:]**2) * lmbda / (2*m)

#vectorization done for regularized gradient
def gradRegularization(theta, X, y, lmbda):
  m = len(y)
  temp = sigmoid(np.dot(X, theta)) - y
  temp = np.dot(temp.T, X).T / m + theta * lmbda / m
  temp[0] = temp[0] - theta[0] * lmbda / m
  return temp

#inital parameters
# lmbda = 0.1 gave accu_perct as 79.38833333333334
# lmbda = 0.2 gave accu_perct as 79.40333333333334
#gave accu_perct as 79.455 test set accuracy as 79.06 with 50 iterations
#gave accu_perct as 79.94 test set accuracy as 79.38 with 100 iterations
#gave accu_perct as 80.366 test set accuracy as 79.43 with 500 iterations, took 1139.07 seconds
#gave accu_perct as 80.4466 test set accuracy as 79.39 with 1000 iterations, took 2304.93 seconds
lmbda = 0.8
k = 10
theta = np.zeros((k,n))

#fmin_cg from scipy library uses cost function minimization
#using a nonlinear conjugate gradient algorithm.
# f is the target of minimization
# fprime is the gradient of taget at the precise values of x and y
# args are the argument list as in the order defined in the functions
# maxiter is the maximum iterations it will run
for i in range(k):
  digit_class = i if i else 10
  theta[i] = opt.fmin_cg(f = costFunctionReg, x0 = theta[i],  fprime = gradRegularization, args = (x, (y == digit_class).flatten(), lmbda), maxiter = 1000)

#predicting as per the model
#using One-vs-All technique
pred = np.argmax(x @ theta.T, axis = 1)
pred = [e if e else 10 for e in pred]
accu_perct = np.mean(pred == y.flatten()) * 100

print(accu_perct)

#testing with test data from mnist_test.csv
#loading the dataset
dataset_test = pd.read_csv('mnist_test.csv')

#specifying predictor and response
# predictor having shape as (10000,784)
x_test = dataset_test.iloc[:, 1:].values 
  
# reponse having shape as (10000,)
y_test = dataset_test.iloc[:, 0].values

#adding the constant term
m = len(y_test)
ones = np.ones((m,1))
x_test = np.hstack((ones, x_test)) #add the constant

#predicting as per the model
#using One-vs-All technique
pred_test = np.argmax(x_test @ theta.T, axis = 1)
pred_test = [e if e else 10 for e in pred_test]
accu_perct_test = np.mean(pred_test == y_test.flatten()) * 100

print(accu_perct_test)

print("Time Taken:",round(time.perf_counter(),2))