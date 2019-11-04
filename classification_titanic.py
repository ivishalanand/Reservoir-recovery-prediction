# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:55:28 2019

@author: vishal
"""


import numpy as np
import pandas as pd
titanic=pd.read_csv("train.csv")

titanic.head()
titanic.info()

titanic.isna().sum()[titanic.isna().sum()>0]

#drop
#name, passenger ID, Ticket, Cabin
titanic=titanic.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)

titanic.head()

#fill na

titanic.isna().sum()[titanic.isna().sum()>0]
titanic.Age=titanic.Age.fillna(titanic.Age.mean())
titanic.isna().sum()[titanic.isna().sum()>0]
titanic.Embarked=titanic.Embarked.fillna(titanic.Embarked.mode())
titanic.isna().sum()[titanic.isna().sum()>0]
titanic.Age.describe()

y=titanic.Survived
X=titanic.drop('Survived',axis=1)
X.head()

X.Age.describe()

#dummies
X=pd.get_dummies(X)
X.columns

#normalise
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X[['Age','Fare']]=scaler.fit_transform(X[['Age','Fare']])

#model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=.2)

from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#performance
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print ("Confusion Matrix : \n", cm) 

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#XGBOOST
pip install xgboost
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

#CATBOOST
pip install catboost
from catboost import CatBoostClassifier
classifier=CatBoostClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#SVM
from sklearn.svm import SVC
classifier=SVC(kernel='poly') #rbg , linear
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

#test
Test= pd.read_csv("test.csv")
Test.info()
Test.isna().sum()
Test.Age=Test.Age.fillna(Test.Age.mean())
Test.Fare=Test.Fare.fillna(Test.Fare.mean())
Test=Test.drop(['Name','PassengerId','Ticket', 'Cabin'],axis=1)

Test=pd.get_dummies(Test)

#normalise
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
Test[['Age','Fare']]=scaler.fit_transform(Test[['Age','Fare']])

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(Test)

#saving
result=pd.DataFrame(y_pred)
cd=pd.concat([Test['PassengerId'],result],axis=1)
cd.columns=['PassengerId','Survived']
result.to_csv("foo2.csv",index=False,header=["Survivor"])
cd.to_csv("ok.csv",index=False)
