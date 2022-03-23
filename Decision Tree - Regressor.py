import pandas as pd
import numpy as np

data = pd.read_csv("Salary.csv")

#encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Position'] = le.fit_transform(data['Position'])

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(data.head)

from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()

kfold = KFold(n_splits=10)
model_eval = cross_val_score(clf, X, y, cv=kfold, scoring='r2')
print("Score:", model_eval)

