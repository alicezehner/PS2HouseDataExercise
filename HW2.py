import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# read in training data
data = pd.read_csv('train.csv')

# select only numeric test data
numeric = data.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)

# correlate data
corr = numeric.corr()['SalePrice']
col = corr.sort_values(ascending=False)

# create X and Y of numeric data and Sale Price to create linReg with
Y = data['SalePrice']
X = numeric.drop(['SalePrice'], axis = 1)

# create lin reg
model = LinearRegression().fit(X, Y)

# predict test data and graph real sale price vs. predicted sale price
predictions = model.predict(X)
#plt.scatter(predictions, Y, color = 'r')

print(f"R^2 value is {model.score(X, Y)}")

test = pd.read_csv('test.csv')
test_num =  test.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)

test_predict = model.predict(test_num)
print(test_predict)