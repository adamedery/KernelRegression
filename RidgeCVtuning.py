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
temp = []
for row in inputData:
    for i in filter(None, re.split("[ ]+", row[0])):
        temp.append(float(i))
    x.append(temp)
    temp = []
for row in targetData:
    for i in filter(None, re.split("[ ]+", row[0])):
        y.append(float(i))


#scaling  and centering the data so it has a mean of 0 and a std dev of 1
scaler = PP.StandardScaler().fit(x)

x = list(scaler.transform(x))

results = []

# use for changeing the neq_mean_squared error into an RMSE
def myConversion(listIN):
    return list(map(lambda x:x**0.5,map(abs,listIN)))

#returns the test and train RMSE of a cross-validation using ridge regression
def modelCalc(x, y, givenAlpha):
    model = LM.Ridge(alpha = givenAlpha)
    scores = MS.cross_validate(model, x, y, cv = 5, scoring = 'neg_mean_squared_error', return_train_score = True)
    return (myConversion(list(scores['test_score'])), myConversion(list(scores['train_score'])))

alphas = [0,0.1,1,10,100,1000,10000,100000]

#runs the above algorithm for all valeus of alpha
for i in alphas:
    results.append(modelCalc(x, y, i))

#function for average
def average(input):
    return float(sum(input)) / len(input)

#outputs all train and test errors and the averages
for i in range(len(results)):
    print('For alpha value of ' + str(alphas[i]) + " the result was:")
    print('Average test error: ' + str(average(results[i][0])))
    print('Average train error: ' + str(average(results[i][1])))
    print('Test errors: ' + str(results[i][0]))
    print('Train errors: ' + str(results[i][1]))
