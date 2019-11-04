# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 18:18:26 2019

@author: vishal
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import skew
from scipy import stats
from scipy.stats.stats import pearsonr
from scipy.stats import norm
from collections import Counter
from sklearn.linear_model import LinearRegression,LassoCV, Ridge, LassoLarsCV,ElasticNetCV
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')
%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook
%matplotlib inline

df = pd.read_csv("processed.csv")
df2=df.sample(frac=1)
#df2.drop("Unnamed: 0",axis=1,inplace=True)
train=df2.iloc[:301,:]
test=df2.iloc[301:,:]

# Check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

# Save the 'Unnamed: 0' column
train_ID = train['Unnamed: 0']
test_ID = test['Unnamed: 0']

# Now drop the 'Id' column since it's unnecessary for the prediction process.
train.drop("Unnamed: 0", axis = 1, inplace = True)
test.drop("Unnamed: 0", axis = 1, inplace = True)

# Check data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))

train.head()
test.head()

# Getting Description
train['REC'].describe()

# Plot Histogram
sns.distplot(train['REC'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['REC'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('REC distribution')

fig = plt.figure()
res = stats.probplot(train['REC'], plot=plt)
plt.show()

print("Skewness: %f" % train['REC'].skew())
print("Kurtosis: %f" % train['REC'].kurt())

#Multivariate Analysis
train.select_dtypes(include=['object']).columns
train.select_dtypes(include=['int64','float64']).columns

cat = len(train.select_dtypes(include=['object']).columns)
num = len(train.select_dtypes(include=['int64','float64']).columns)
print('Total Features: ', cat, 'categorical', '+',
      num, 'numerical', '=', cat+num, 'features')

# Correlation Matrix Heatmap
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 15 #number of variables for heatmap
cols = corrmat.nlargest(k, 'REC')['REC'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


#OOIP vs REC
sns.jointplot(x=train['OOIP'], y=train['REC'], kind='reg')
#removing the outliers manually
train = train.drop(train[(train['OOIP']>2) & (train['REC']>1250)].index).reset_index(drop=True)
sns.jointplot(x=train['OOIP'], y=train['REC'], kind='reg')

#K vs REC
sns.jointplot(x=train['K'], y=train['REC'], kind='reg')
#removing the outlier
train = train.drop(train[(train['K']>15) & (train['REC']>0)].index).reset_index(drop=True)
sns.jointplot(x=train['K'], y=train['REC'], kind='reg')


#k/uob vs REC
sns.jointplot(x=train['k/uob'], y=train['REC'], kind='reg')
#removing outliers
train = train.drop(train[(train['k/uob']>10) & (train['REC']>0)].index).reset_index(drop=True)
sns.jointplot(x=train['k/uob'], y=train['REC'], kind='reg')


#pep vs REC
sns.jointplot(x=train['Pep'], y=train['REC'], kind='reg')


#T vs REC
sns.jointplot(x=train['T'], y=train['REC'], kind='reg')
#removing outliers
train = train.drop(train[(train['T']>10) & (train['REC']>0)].index).reset_index(drop=True)
sns.jointplot(x=train['k/uob'], y=train['REC'], kind='reg')

#RSA vs REC
sns.jointplot(x=train['Rsa'], y=train['REC'], kind='reg')


#Uoi vs REC
sns.jointplot(x=train['Uoi'], y=train['REC'], kind='reg')

#Uoa vs REC
sns.jointplot(x=train['Uoa'], y=train['REC'], kind='reg')

#Bp vs REC
sns.jointplot(x=train['Bp'], y=train['REC'], kind='reg')
#removing outliers
train = train.drop(train[(train['Bp']>15) & (train['REC']>0)].index).reset_index(drop=True)
sns.jointplot(x=train['Bp'], y=train['REC'], kind='reg')

#Ubp vs REC
sns.jointplot(x=train['Ubp'], y=train['REC'], kind='reg')


#h vs REC
sns.jointplot(x=train['h'], y=train['REC'], kind='reg')
#removing outliers
train = train.drop(train[(train['h']>10) & (train['REC']>0)].index).reset_index(drop=True)
sns.jointplot(x=train['h'], y=train['REC'], kind='reg')

#Porosity vs REC
sns.jointplot(x=train['POROSITY'], y=train['REC'], kind='reg')
#removing outliers

#Boa vs REC
sns.jointplot(x=train['Boa'], y=train['REC'], kind='reg')


# Combining Datasets
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.REC.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['REC'], axis=1, inplace=True)
print("Train data size is : {}".format(train.shape))
print("Test data size is : {}".format(test.shape))
print("Combined dataset size is : {}".format(all_data.shape))



y_train = train.REC.values

print("Skewness: %f" % train['REC'].skew())
print("Kurtosis: %f" % train['REC'].kurt())


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index


train = all_data[:ntrain]
test = all_data[ntrain:]


#Modelling and predictions
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Cross-validation with k-folds
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.0, 
                             learning_rate=0.05, max_depth=6, 
                             min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                             random_state =7)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


#Checking performance of base models by evaluating the cross-validation RMSLE error.

score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))



#Stack models to average their score
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
    
#Here we average ENet, GBoost, KRR, and lasso. We'll add in XGBoost and LightGBM later.

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(clf)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


#Since our Gradient boosting score model performed the best, we'll use it as a meta-model.

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = model_xgb)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*0.70 +
               xgb_train_pred*0.10 + lgb_train_pred*0.20 ))

Stacked = 1/(0.1077)
XGBoost = 1/(0.1177)
LGBM = 1/(0.1159)
Sum = Stacked + XGBoost + LGBM
Stacked = Stacked/Sum
XGBoost = XGBoost/Sum
LGBM = LGBM/Sum
print(Stacked, XGBoost, LGBM)


'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train,stacked_train_pred*Stacked +
               xgb_train_pred*XGBoost + lgb_train_pred*LGBM))

ensemble = stacked_pred*Stacked + xgb_pre220d*XGBoost + lgb_pred*LGBM

