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

inputData = csv.reader(open("Datasets/hw1-q1x.csv", "r"))
targetData = csv.reader(open("Datasets/hw1-q1y.csv", "r"))

x = []
y = []
temp = []
for row in inputData:
    for i in filter(None, re.split("[ ]+", row[0])):
        temp.append(float(i))
    x.append(temp)
    temp = []

for row in targetData:
    for i in filter(None, re.split("[ ]+", row[0])):
        y.append(float(i))

indexing = list(range(len(x)))
shuffle(indexing)

x_test = []
y_test = []
x_train = []
y_train = []

counter = 1
for i in indexing:
    if counter % 5 == 0:
        x_test.append(x[i])
        y_test.append(y[i])
        counter = 1
    else:
        x_train.append(x[i])
        y_train.append(y[i])
        counter += 1

#scaling  and centering the data so it has a mean of 0 and a std dev of 1
scaler = PP.StandardScaler().fit(x_train)

x_train_temp = scaler.transform(x_train)

x_test_temp = scaler.transform(x_test)

x_train_s = []
x_test_s = []
#putting the data into amore usable type for use by other functions
for i in range(len(x_train_temp)):
    x_train_s.append(x_train_temp[i][1])

for i in range(len(x_test_temp)):
    x_test_s.append(x_test_temp[i][1])

#returns the list of powers up to d of both attributes in each array element
def kernelD(inputList, d):
    return list(map(lambda x : [x ** i for i in range(d)], inputList))

# use for changing the neq_mean_squared error into an RMSE
def myConversion(listIN):
    return list(map(lambda x : x ** 0.5, map(abs, listIN)))

# function for average
def average(input):
    return float(sum(input)) / len(input)

# returns the test RMSE of a cross-validation using Lasso regression
def modelCalcL1(x, y, givenAlpha):
    model = LM.Lasso(alpha = givenAlpha, max_iter = 100000)
    scores = MS.cross_validate(model, x, y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = False)
    predictions = MS.cross_val_predict(model, x, y, cv = 5)
    model.fit(x,y)
    return (predictions, average(myConversion(list(scores['test_score']))), list(model.coef_))

# returns the test RMSE of a cross-validation using ridge regression
def modelCalcL2(x, y, givenAlpha):
    model = LM.Ridge(alpha = givenAlpha)
    scores = MS.cross_validate(model, x, y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = False)
    predictions = MS.cross_val_predict(model, x, y, cv = 5)
    model.fit(x,y)
    return (predictions, average(myConversion(list(scores['test_score']))), list(model.coef_))

resultsL1 = []
alphas = [0.01,0.1,1,10]
for i in alphas:
    resultsL1.append(modelCalcL1(kernelD(x_train_s, 9), y_train, i))

for i in range(len(alphas)):
    print('Lasso RMSE for alpha ' + str(alphas[i]) + ': ' + str(resultsL1[i][1]))
print(min(i[1] for i in resultsL1))
print([resultsL1[i][2] for i in range(len(resultsL1))])

plt.figure(1)
plt.plot(x_train_s, resultsL1[0][0], 'ro', markersize=1)
plt.plot(x_train_s, resultsL1[1][0], 'bo', markersize=1)
plt.plot(x_train_s, resultsL1[2][0], 'yo', markersize=1)
plt.plot(x_train_s, resultsL1[3][0], 'go', markersize=1)
plt.plot(x_train_s, y_train, 'ko', markersize=1)
plt.axis([-2, 2, -50, 150])
plt.ylabel('Predicted value')
plt.xlabel('Input value')
plt.title('Lasso predictions using differnet alpha')

resultsL2 = []
for i in alphas:
    resultsL2.append(modelCalcL2(kernelD(x_train_s, 9), y_train, i))

for i in range(len(alphas)):
    print('Ridge RMSE for alpha ' + str(alphas[i]) + ': ' + str(resultsL2[i][1]))
print(min(i[1] for i in resultsL2))
print([resultsL2[i][2] for i in range(len(resultsL2))])

plt.figure(2)
plt.plot(x_train_s, resultsL2[0][0], 'ro', markersize=1)
plt.plot(x_train_s, resultsL2[1][0], 'bo', markersize=1)
plt.plot(x_train_s, resultsL2[2][0], 'yo', markersize=1)
plt.plot(x_train_s, resultsL2[3][0], 'go', markersize=1)
plt.plot(x_train_s, y_train, 'ko', markersize=1)
plt.axis([-2, 2, -50, 150])
plt.ylabel('Predicted value')
plt.xlabel('Input value')
plt.title('Ridge predictions using differnet alpha')
plt.show()
