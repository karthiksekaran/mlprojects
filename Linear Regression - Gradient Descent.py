import numpy as np
import pandas as pd
class GradientDescentLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate, self.iterations = learning_rate, iterations
    
    def fit(self, X, y):
        b = 0
        m = 5
        n = X.shape[0]
        for _ in range(self.iterations):
            b_gradient = -2 * np.sum(y - m*X + b) / n
            m_gradient = -2 * np.sum(X*(y - (m*X + b))) / n
            b = b + (self.learning_rate * b_gradient)
            m = m - (self.learning_rate * m_gradient)
        self.m, self.b = m, b
        
    def predict(self, X):
        return self.m*X + self.b

np.random.seed(42)
X = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5)
y = np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25)
"""
data = pd.read_csv("height-weight.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
"""
clf = GradientDescentLinearRegression()
clf.fit(X, y)
"""
import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')

plt.scatter(X, y, color='black')
plt.plot(X, clf.predict(X))
plt.gca().set_title("Gradient Descent Linear Regressor")
"""
clf.b
clf.m
print("Result:")
pred = clf.predict(X)
print("y:", y)
print("y_pred:", pred)
from sklearn.metrics import r2_score
scores = r2_score(y, pred)
print("Scores:",scores*100)
