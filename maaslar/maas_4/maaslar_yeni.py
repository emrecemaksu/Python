#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 00:29:27 2018

@author: emrecemaksu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm

maaslar_yeni = pd.read_csv('maaslar_yeni.csv')

unvan_seviyesi = maaslar_yeni[['UnvanSeviyesi']]
Kidem = maaslar_yeni[['Kidem']]
Puan = maaslar_yeni[['Puan']]
maas = maaslar_yeni[['maas']].values
veriler = pd.concat([unvan_seviyesi, Kidem, Puan,], axis=1).values
veriler2 = pd.concat([unvan_seviyesi, Kidem], axis=1).values
veriler3 = pd.concat([unvan_seviyesi], axis=1).values

#LinearRegression

LR = LinearRegression()
LR = LR.fit(veriler3, maas)
model = sm.OLS(LR.predict(veriler3), veriler3)
print(model.fit().summary())
#PolynomialRegression

PR = PolynomialFeatures(degree=2)
PR_veriler = PR.fit_transform(veriler)
LR2 = LinearRegression().fit(PR_veriler, maas)
PR_predict = LR2.predict(PR_veriler)

model2 = sm.OLS(PR_predict, veriler)
print(model2.fit().summary())

#SVR

svr = SVR(kernel='rbf')
svr.fit(veriler, maas)
SVR_predict = svr.predict(veriler)
model3 = sm.OLS(SVR_predict, maas)
print(model3.fit().summary())

#TreeRegressor 

DTR = DecisionTreeRegressor(random_state=0)
SC = StandardScaler()
veriler_SC = SC.fit_transform(veriler)
veriler2_SC = SC.fit_transform(veriler2)
veriler3_SC = SC.fit_transform(veriler3)
SC_maas = SC.fit_transform(maas)
DTR.fit(veriler_SC, SC_maas)
DTR_predict = DTR.predict(veriler_SC)
model4 = sm.OLS(DTR_predict, SC_maas)
print(model4.fit().summary())

#RandomForestRegressor

RFR = RandomForestRegressor(n_estimators=10, random_state=0)
RFR.fit(veriler, maas)
RFR_predict = RFR.predict(veriler)
model5 = sm.OLS(RFR_predict, maas)
print(model5.fit().summary())
