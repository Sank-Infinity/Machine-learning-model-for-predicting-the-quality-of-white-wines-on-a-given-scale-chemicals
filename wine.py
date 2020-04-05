# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:08:12 2020

@author: Sanket Kale
"""
#Importing libraries
import numpy as np
import pandas as pd

#Importing Dataset
dataset = pd.read_csv('wine.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 11].values

#Split dataset into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Importing regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state =0 )
regressor.fit(X_train, Y_train)

#predicting results
Y_pred = regressor.predict(X_test)


#Backword elimination 
import statsmodels.formula.api as sm

#1st iteration
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()

#2nd iteration
X_opt = X[:, [0,1,3,5,7,9,10]]
regressor_OLS = sm.OLS(endog= Y, exog= X_opt).fit()
regressor_OLS.summary()

#spliting optimal dataset as training and test set 
from sklearn.cross_validation import train_test_split
X_train_opt , X_test_opt, Y_train_opt, Y_test_opt = train_test_split(X_opt, Y, test_size=0.2, random_state= 0)

#Initialising regressor for optimal dataset
regressor.fit(X_train_opt, Y_train_opt)

#predicting optimal results 
Y_pred_opt = regressor.predict(X_test_opt)
