import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('logit-data.txt', header=None)
df.head()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

#Add a bias column to the X. The value of the bias column is usually one.

X = np.c_[np.ones((X.shape[0], 1)), X]
X[:10]

#Here, our X is a two-dimensional array and y is a one-dimensional array. Let’s make the ‘y’ two-dimensional to match the dimensions.

y = y[:, np.newaxis]
y[:10]

def sigmoid(x, theta):
    z= np.dot(x, theta)
    return 1/(1+np.exp(-z))

def hypothesis(theta, x):
    return sigmoid(x, theta)

def cost_function(theta, x, y):
    m = X.shape[0]
    h = hypothesis(theta, x)
    return -(1/m)*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def gradient(theta, x, y):
    m = X.shape[0]
    h = hypothesis(theta, x)
    return (1/m) * np.dot(X.T, (h-y))

theta = np.zeros((X.shape[1], 1))
from scipy.optimize import minimize,fmin_tnc
def fit(x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(x, y.flatten()))
    return opt_weights[0]
parameters = fit(X, y, theta)

h = hypothesis(parameters, X)

def predict(h):
    h1 = []
    for i in h:
        if i>=0.5:
            h1.append(1)
        else:
            h1.append(0)
    return h1
y_pred = predict(h)

accuracy = 0
for i in range(0, len(y_pred)):
    if y_pred[i] == y[i]:
        accuracy += 1
accuracy/len(y)
print(accuracy)