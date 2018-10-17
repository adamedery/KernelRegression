import csv
import numpy as np
import scipy
from sklearn import preprocessing as PP
from sklearn import linear_model as LM
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
temp = []
for row in inputData:
    for i in filter(None, re.split("[ ]+", row[0])):
        temp.append(float(i))
    x.append(temp)
    temp = []
for row in targetData:
    for i in filter(None, re.split("[ ]+", row[0])):
        y.append(float(i))

#shuffle an indexing of the data and using it to randomly split the data
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

x_train_s = list(scaler.transform(x_train))

x_test_s = list(scaler.transform(x_test))

results = []

#creating the model and testing the accuracy at each point in the test set
#function returns the RMSE of the test set, the L2 norm of the weight vector, and the values of the weights
def modelCalc(x_train, y_train, x_test, y_test, givenAlpha):
    model = LM.Ridge(alpha = givenAlpha)
    model.fit(x_train, y_train)
    y_predict_test = model.predict(x_test)
    sumErrorTest = 0
    for i in range(len(y_predict_test)):
        sumErrorTest += (y_predict_test[i] - y_test[i]) ** 2
    y_predict_train = model.predict(x_train)
    sumErrorTrain = 0
    for i in range(len(y_predict_train)):
        sumErrorTrain += (y_predict_train[i] - y_train[i]) ** 2
    return [(sumErrorTrain / len(y_predict_train)) ** 0.5, (sumErrorTest / len(y_predict_test)) ** 0.5, sum(list(map(lambda x:x**2,model.coef_)))**0.5, list(model.coef_)]

alphas = [0,0.1,1,10,100,1000,10000,100000]

#running the model builder for all values of alpha
for i in alphas:
    results.append(modelCalc(x_train_s, y_train, x_test_s, y_test, i))

#moving the results into more descriptive lists so the next section is easier to understand for the graders
res_RMSE_test = []
res_RMSE_train = []
res_L2 = []
res_weights1 = []
res_weights2 = []
for i in range(len(results)):
    res_RMSE_train.append(results[i][0])
    res_RMSE_test.append(results[i][1])
    res_L2.append(results[i][2])
    res_weights1.append(results[i][3][0])
    res_weights2.append(results[i][3][1])

#plotting the RMSE on the test and train set for each value of alpha
plt.figure(1)
plt.plot([-2,-1,0,1,2,3,4,5], res_RMSE_test, 'bo')
plt.plot([-2,-1,0,1,2,3,4,5], res_RMSE_train, 'ro')
plt.axis([-3, 6, 0, max(res_RMSE_test) + 5])
plt.xlabel('log(alpha)')
plt.ylabel('RMSE')
plt.title('RMSE vs alpha')

#plotting the L2 norm of the weights for each value of alpha
plt.figure(2)
plt.plot([-2,-1,0,1,2,3,4,5], res_L2, 'ro')
plt.axis([-3, 6, 0, max(res_L2) + 5])
plt.xlabel('log(alpha)')
plt.ylabel('L2 Norm of weights')
plt.title('L2 Norm vs alpha')

#plotting the values of the weights for each value of alpha
plt.figure(3)
plt.plot([-2,-1,0,1,2,3,4,5], res_weights1, 'ro')
plt.plot([-2,-1,0,1,2,3,4,5], res_weights2, 'bo')
plt.axis([-3, 6, 0, max(max(res_weights1),max(res_weights2)) + 5])
plt.xlabel('log(alpha)')
plt.ylabel('Weights')
plt.title('Weights vs alpha')

plt.show()
