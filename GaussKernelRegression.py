import csv
import numpy as np
import scipy
from sklearn import preprocessing as PP
from sklearn import linear_model as LM
from sklearn import model_selection as MS
import matplotlib.pyplot as plt
import re
from random import shuffle
import math

#reading the data from the input files
inputData = csv.reader(open("Datasets/hw1-q1x.csv", "r"))
targetData = csv.reader(open("Datasets/hw1-q1y.csv", "r"))

#splitting the data and filtering the whitespace
x = []
y = []
for row in inputData:
    temp = []
    for i in filter(None, re.split("[ ]+", row[0])):
        temp.append(float(i))
    x.append(temp)

for row in targetData:
    for i in filter(None, re.split("[ ]+", row[0])):
        y.append(float(i))

#scaling  and centering the data so it has a mean of 0 and a std dev of 1
scaler = PP.StandardScaler().fit(x)

x = list(scaler.transform(x))

#computes the gaussian of a given value centered at miu and with std dev of sigma
def gaussBasis(value, miu, sigma):
    return np.exp(((value - miu) ** 2) / (-2 * sigma))

transX = [[],[],[],[]]

miuSet = [-1,-0.5,0,0.5,1]
varSet = [0.1,0.5,1,5]

# puts the input data through the different gaussian kernels and stores the results
# uses the variance and means of the above arrays
for i in range(len(varSet)):
    for val in x:
        temp = []
        for miu in miuSet:
            temp.append(gaussBasis(val[0], miu, varSet[i]))
            temp.append(gaussBasis(val[1], miu, varSet[i]))
        transX[i].append(temp)

# use for changing the neq_mean_squared error into an RMSE
def myConversion(listIN):
    return list(map(lambda x : x ** 0.5, map(abs, listIN)))

# function for average
def average(input):
    return float(sum(input)) / len(input)

# returns the test and train RMSE of a cross-validation using linear regression
def modelCalc(x, y):
    model = LM.LinearRegression()
    scores = MS.cross_validate(model, x, y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
    return (average(myConversion(list(scores['test_score']))), average(myConversion(list(scores['train_score']))))

# compute all of the test and train RMSE for the different kernels
results = []
for set in transX:
    results.append(modelCalc(set, y))

plt.figure(1)
plt.plot(varSet, [i[0] for i in results], 'ro')
plt.plot(varSet, [i[1] for i in results], 'bo')
plt.plot([0,5], [15.2,15.2], 'y-')
plt.axis([0,5, 0, 40])
plt.xlabel('variance')
plt.ylabel('RMSE')
plt.title('test/train error vs alpha')

'''
plt.figure(2)
plt.plot(varSet, [i[0] for i in results[4:8]], 'ro')
plt.plot(varSet, [i[1] for i in results[4:8]], 'bo')
plt.plot([0,5], [15.2,15.2], 'y-')
plt.axis([0,5, 0, 40])
plt.xlabel('variance')
plt.ylabel('miu = -0.5')
plt.title('test/train error vs alpha')

plt.figure(3)
plt.plot(varSet, [i[0] for i in results[8:12]], 'ro')
plt.plot(varSet, [i[1] for i in results[8:12]], 'bo')
plt.plot([0,5], [15.2,15.2], 'y-')
plt.axis([0,5, 0, 40])
plt.xlabel('variance')
plt.ylabel('miu = 0')
plt.title('test/train error vs alpha')

plt.figure(4)
plt.plot(varSet, [i[0] for i in results[12:16]], 'ro')
plt.plot(varSet, [i[1] for i in results[12:16]], 'bo')
plt.plot([0,5], [15.2,15.2], 'y-')
plt.axis([0,5, 0, 40])
plt.xlabel('variance')
plt.ylabel('miu = 0.5')
plt.title('test/train error vs alpha')

plt.figure(5)
plt.plot(varSet, [i[0] for i in results[16:20]], 'ro')
plt.plot(varSet, [i[1] for i in results[16:20]], 'bo')
plt.plot([0,5], [15.2,15.2], 'y-')
plt.axis([0,5, 0, 40])
plt.xlabel('variance')
plt.ylabel('miu = 1')
plt.title('test/train error vs alpha')
'''

plt.show()
