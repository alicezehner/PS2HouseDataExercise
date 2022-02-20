import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')
data.head()

filtered = data.select_dtypes(include=[np.number])
filtered.head()

data.corr()['SalePrice']

