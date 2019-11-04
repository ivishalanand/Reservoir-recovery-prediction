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
import numpy as np
X=pd.read_csv("rrf.csv")


np.random.seed(42)
X.describe()
X.info()

X.isna().any()

y=pd.DataFrame(X.REC)
X=X.drop(["REC"],axis=1)

X['k/uob'] =pd.to_numeric(X['k/uob'],errors='coerce')

# ================================================================.=============
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


import matplotlib.pyplot as plt
import seaborn as sns

#removing anomaly
from sklearn.ensemble import IsolationForest

clf = IsolationForest(contamination=0.05, random_state=42)
clf.fit(X_numeric)
anomaly=clf.predict(X_numeric)
X_numeric['anomaly']=anomaly
X_categorical['ANOMALY']=anomaly
y['anomaly']=anomaly
X_numeric=X_numeric[X_numeric.anomaly>0]
X_categorical=X_categorical[X_categorical.ANOMALY>0]
y=y[y.anomaly>0]


#feature selection
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X=pd.concat([X_numeric,X_categorical],axis=1)
X.isna().any()
X=X.drop(['anomaly','ANOMALY'],axis=1)
y=y.drop(['anomaly'],axis=1)

#feature selection



#apply SelectKBest class to extract top 10 best features
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')
plt.show()

selected_features_15=['OOIP','POROSITY','K','Sw','T','Rsb','Pep','k/uob','Bol','Rsa','PI','Boa','Uw','Uoa','API',1,2,3,4]
selected_features_10=['OOIP','POROSITY','K','Sw','T','Rsb','Pep','k/uob','Bol','Rsa',1,2,3,4]





#heatmap

import seaborn as sns
corrmat = X.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# =============================================================================
# FROM HERE YOUR PREVIOUS WORK
# =============================================================================





#15 features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[selected_features_15],y,test_size=.2)







#PCA
a=[]
for n in range(2,19):
    
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features_15],y,test_size=.2)
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    X_train = pca.fit_transform(X_train) 
    X_test = pca.transform(X_test) 
    explained_variance = pca.explained_variance_ratio_
    
    classifier=LinearRegression()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test) 
    
    a.append( mean_squared_error(y_pred,y_test))
    print()

#plotting mse wrt pca dimention 
plt.plot(a)
#best dimention is 9 dimention using pca

X=X.reset_index().drop(['index'],axis=1)
y=y.reset_index().drop(['index'],axis=1)


# =============================================================================
# #cross validation
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=3)
# classifier=LinearRegression()
# for train_index, test_index in kf.split(X):
#     print("Train:", train_index, "Validation:",test_index)
#     print(X_15.loc[train_indices], y.loc[train_indices]
#     print(classifier.score(X[test_indices], y[test_indices]))
# 
# =============================================================================

# Linear Regression model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression 
clf_lr = LinearRegression()
clf_lr.fit(X_train,y_train)
y_pred_lr = clf_lr.predict(X_test)
mean_squared_error(y_pred_svr,y_test)


from sklearn.svm import SVR
clf_svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
clf_svr.fit(X_train,y_train)
y_pred_svr = clf_svr.predict(X_test)
mean_squared_error(y_pred_svr,y_test)

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
clf_rf = RandomForestRegressor(n_estimators=100)
clf_rf.fit(X_train,y_train)
y_pred_rf = clf_rf.predict(X_test)
mean_squared_error(y_pred_rf,y_test)

from sklearn.ensemble import GradientBoostingRegressor
clf_gb = GradientBoostingRegressor(n_estimators=200)
clf_gb.fit(X_train,y_train)
y_pred_gb = clf_gb.predict(X_test)
mean_squared_error(y_pred_gb,y_test)

a=X
a['REC']=y

a.to_csv("processed.csv")