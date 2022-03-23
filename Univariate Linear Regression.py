#Univariate Linear Regression

import numpy as np
import pandas as pd
df=pd.read_csv("height-weight.csv")
df.head()

df.columns
df.shape
df.isna().any()

df.corr()

height=df.Height.values[:,np.newaxis]
weight=df.Weight.values
height
weight

#Formula:- Xnormal=(X-Xmin)/(Xmax-Xmin), where X is the values, Xman is the maximum value of the X and Xmin is the minimum value of this X.

Heightmin=height.min()
Heightmax=height.max() 
Heightnorm=(height-Heightmin)/(Heightmax-Heightmin)
Weightmin=weight.min()
Weightmax=weight.max()
Weightnorm=(weight-Weightmin)/(Weightmax-Weightmin)
Heightnorm
Weightnorm

import sklearn.linear_model as lm 
lr=lm.LinearRegression() 
lr.fit(height,weight)

knownvalue=float(input("Enter the value of height:"))
findvalue=lr.predict([[knownvalue]])
print("when the height value is",knownvalue,"that moment weight value is",findvalue)

df["predicted_value"]=lr.predict(height)
df.head()

from sklearn.metrics import r2_score
accuracy=r2_score(weight,lr.predict(height))
print("the model accuracy is",accuracy*100,"%")