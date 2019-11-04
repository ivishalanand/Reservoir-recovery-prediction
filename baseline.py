# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 21:45:42 2019

@author: vishal
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:54:28 2019

@author: vishal
"""

import pandas as pd
from math import sqrt
import numpy as np
X=pd.read_csv("okay.csv",error_bad_lines=False, engine='python')


np.random.seed(42)
X.describe()
X.info()

X.isna().any()

y=X.REC
X=X.drop(["REC"],axis=1)

X['k/uob'] =pd.to_numeric(X['k/uob'],errors='coerce')



# =============================================================================
# X=X.drop(columns=["k/uob"],axis=1)
# =============================================================================
X.isna().any()
X["k/uob"].fillna(X["k/uob"].mean(), inplace=True)
X.isna().any()

X_categorical=pd.get_dummies(X.Type)
X_numeric=X.drop(["Type"],axis=1)

X.info()



from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_numeric = pd.DataFrame(scaler.fit_transform(X_numeric), index=X_numeric.index, columns=X_numeric.columns)




X=pd.concat([X_numeric,X_categorical],axis=1)
X.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)



from sklearn.linear_model import LinearRegression
classifier=LinearRegression()
classifier.fit(X_train,y_train)

y_pred=pd.DataFrame(classifier.predict(X_test))

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error  
print(mean_squared_error(y_pred,y_test))
print(mean_absolute_error(y_pred,y_test))
print(sqrt(mean_squared_error(y_pred,y_test)))
print(r2_score(y_pred,y_test))
print(median_absolute_error (y_pred,y_test))


