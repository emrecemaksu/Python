#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 20:16:20 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

tenis = pd.read_csv('odev_tenis.csv')
windy = tenis[['windy']]
play = tenis[['play']]
temphumi = tenis[['temperature', 'humidity']]
outlook = tenis[['outlook']].values
humidity = tenis[['humidity']]


LE = LabelEncoder()
windy = LE.fit_transform(windy)
play = LE.fit_transform(play)
outlook[:,0] = LE.fit_transform(outlook[:,0])

OHE = OneHotEncoder(categorical_features="all")
outlook = OHE.fit_transform(outlook).toarray()

windy = pd.DataFrame(data = windy, index = range(14), columns=['windy'])
outlook = pd.DataFrame(data = outlook, index = range(14), columns=['overcast', 'rainy', 'sunny'])
play = pd.DataFrame(data = play, index = range(14), columns=['play'])
veriler = pd.concat([outlook, temphumi, windy], axis=1)

x_train, x_test, y_train, y_test = train_test_split(veriler, play, test_size=0.33, random_state=0)

LR = LinearRegression()
LR = LR.fit(x_train, y_train)
predict = LR.predict(x_test)
predict2 = LR.predict(veriler)

bir = np.append(arr = np.ones((14,1)).astype(int), values=veriler, axis=1)
bir = pd.DataFrame(data = bir, index=range(14), columns=['sabit', 'overcast', 'rainy', 'sunny', 'temperature', 'humidity', 'windy'])
bir = bir.iloc[:, [0,1,2,3,4,5,6]].values
r_ols = sm.OLS(endog=play, exog=bir).fit()
print(r_ols.summary())

#tenis = tenis.apply(LabelEncoder.fit_transform)

bir = pd.DataFrame(data = bir, index=range(14), columns=['sabit', 'overcast', 'rainy', 'sunny', 'temperature', 'humidity', 'windy'])
bir = bir.iloc[:, [0,1,2,3,5,6]].values
r_ols = sm.OLS(endog=play, exog=bir).fit()
print(r_ols.summary())

antitemp = pd.concat([outlook, humidity, windy], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(antitemp, play, test_size=0.33, random_state=0)
LR2 = LinearRegression()
LR2 = LR2.fit(X_train, Y_train)
predict2 = LR2.predict(X_test)
