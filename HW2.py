import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# uncomment to graph linreg
#import matplotlib.pyplot as plt

# read in training data
data = pd.read_csv('train.csv')

# select only numeric test data
valid = data.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)
# select first 1300 to make ligreg out of
numeric = valid.head(1300)

# uncomment to view most correlated data
#corr = numeric.corr()['SalePrice']
#col = corr.sort_values(ascending=False)

# create X and Y of numeric data and Sale Price to create linReg with
Y = numeric['SalePrice']
X = numeric.drop(['SalePrice'], axis = 1)

# create lin reg
model = LinearRegression().fit(X, Y)

# uncomment to predict test data and graph real sale price vs. predicted sale price
#predictions = model.predict(X)
#plt.scatter(predictions, Y, color = 'r')

# test prediction validity on rest of training data - select end
M = valid.tail(100)
train_test = M.drop(['SalePrice'], axis = 1)
# use model on train_test
train_predict = model.predict(train_test)

# scatterplot confirms validity of model. Uncomment to view plot
#plt.scatter(M['SalePrice'], train_predict, color = 'r')

# read in test data and select only numeric data
test = pd.read_csv('test.csv')
test_num =  test.select_dtypes(include=[np.number]).interpolate().dropna(axis=1)


# create list of predicted sale prices
test_predict = model.predict(test_num)
# Round prices to the nearest cent
testPredictRounded = np.around(test_predict, decimals=2)
# add predicted price to test table
test['SalePrice'] = testPredictRounded

finalOutput = test[['Id', 'SalePrice']]
#print predicted price and Id for user
print(finalOutput.to_string(index=False))

#export final output to a .csv
finalOutput.to_csv (r'predictions.csv', index=False, header=True)