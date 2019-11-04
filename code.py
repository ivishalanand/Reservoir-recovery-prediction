# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:00:58 2019

@author: vishal
"""

%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rrf=pd.read_csv(r"RRF.csv")
rrf=rrf.convert_objects(convert_numeric=True)
rrf=rrf.dropna(axis='rows')

y=rrf['REC']
rrf=rrf.drop(columns='REC')
rrf_dummy = pd.get_dummies(rrf['Type'])
rrf=rrf.join(rrf_dummy)
X=rrf.drop(columns='Type')

X=X.convert_objects(convert_numeric=True)
X.dtypes
X=X.dropna(axis='rows')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=101)

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train,y_train)
y_pred = regr.predict(X_test)


from sklearn.metrics import mean_squared_error, r2_score

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('R2 score: %.2f' % r2_score(y_test, y_pred))

