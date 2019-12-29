# MNIST_approaches
This is a collection of approaches to recognize handwritten digits for MNIST data obtained from [kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv)

The MNIST dataset provided in a easy-to-use CSV format
The original dataset is in a format that is difficult for beginners to use. This dataset uses the work of Joseph Redmon to provide the MNIST dataset in a CSV format.

## The dataset consists of two files:

- mnist_train.csv
- mnist_test.csv

The mnist_train.csv file contains the 60,000 training examples and labels. The mnist_test.csv contains 10,000 test examples and labels.
Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

This reository has been created solely for the training and exploratory purpose. While I was studying about the algorithms, I have often found it confusing and challenging to choose an approach. That's why I have created to this repository so as to compare and explore the nuances of different techniques.

## Here are the approaches taken:

- [x] logistic_app1.py : Implementing logistic regression algorithms and trying with different lambda to observe the changes in accuracy
- [x] knn_app2.py : Implementing classification algorithm (K Nearest Neighbours) and trying to acheive best accuracy with changing the number of clusters
- [x] svm_app3.py : Implementing Suport Vector Machines with polynomial kernel and choosing gamma with accuracy
- [ ] rfc_app4.py : Implementing Random Forrest classifier
- [ ] cnn_app4.py : Implementing Convoluted Neural Networks
- [ ] deep dive into further advanced aproaches

All the time figures have been calculated according to my presonal computer having the following specification and I have used Sublime Text 3 as the ide for the python programs:

- OS Name -	Microsoft Windows 8.1 Single Language
- Version -	6.3.9600 Build 9600
- Processor - Intel(R) Core(TM) i3-4010U CPU @ 1.70GHz, 1701 Mhz, 2 Core(s), 4 Logical Processor(s)
- Installed Physical Memory (RAM) -	4.00 GB
