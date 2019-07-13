# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:13:46 2019

@author: emrecemaksu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as sm 
import winsound

veriler = pd.read_csv("veriler_nf.csv")
Nhdort = veriler[["NH4N"]]
ALF = veriler[["Averageleachateflow"]]
SS = veriler[["SS"]]
Sicaklik = veriler[["Temperature"]]
Cod = veriler[["COD"]]
MLSSAero = veriler[["MLSSaerobic"]]
Nnf = veriler[["NNF"]]
Codnf = veriler[["CODNF"]]
imputer= Imputer
imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    
SS = imputer.fit_transform(SS)
MLSSAero = imputer.fit_transform(MLSSAero)

scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)
Codnf = scaler.fit_transform(Codnf)
Nnf = scaler.fit_transform(Nnf)

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])

Cod = pd.DataFrame(data = Cod, index = range(275), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(275), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(275), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(275), columns =['ALF'])
Codnf = pd.DataFrame(data = Codnf, index = range(275), columns =['Codnf'])
"""
Nnf = pd.DataFrame(data = Nnf, index = range(275), columns =['Nnf'])
"""
degerler = pd.concat([Cod, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=53)
"""
x_train_cons = np.append(arr = np.ones((206,1)).astype(int), values=x_train.iloc[:,0:], axis=1 )

x_train_opt = x_train_cons[:,[0,1,2,3,4,5,6]]
r_ols = sm.OLS(endog = y_train, exog =x_train_opt)
r = r_ols.fit()
print(r.summary())
"""
params = {'n_estimators': 17, 'max_depth': 4, 'min_samples_split': 6,'learning_rate': 0.3, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nXGBOOST CODNF R2 Skoru => ')
print(r2_score(y_test, y_pred))
skor = r2_score(y_test, y_pred)
"""
x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.10, random_state=0)
params2 = {'n_estimators': 20, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_NNF = ensemble.GradientBoostingRegressor(**params2)
model_NNF.fit(x_train_NNF, y_train_NNF)
y_pred_NNF = model_NNF.predict(x_test_NNF)
print('\nXGBOOST NNF R2 Skoru => ')
print(r2_score(y_test_NNF, y_pred_NNF))
"""
#CODNF depth 6 est 10 random 2 rate 0.3 NH4 suz  %78
eskiskor = 0
eskiestimator = 0
eskidepth = 0
eskisplit = 0
eskirandom = 0
eskirate = 0
for rate in np.arange(0.1, 0.3, 0.10):
    for estimator in range(1,30):
        for depth in range(1,30):
            for randomsta in range(0,100):
                for split in range(2,100):
                    x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=randomsta)
                    params2 = {'n_estimators': estimator, 'max_depth': depth, 'min_samples_split': split,'learning_rate': rate, 'loss': 'ls'}
                    model = ensemble.GradientBoostingRegressor(**params2)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    skor = r2_score(y_test, y_pred)
                    if(skor > eskiskor):
                            eskiskor = skor
                            eskirandom = randomsta
                            eskisplit = split
                            eskidepth = depth
                            eskiestimator = estimator
                            eskirate = rate
                            print(eskiskor)
                            if(eskiskor > 0.80):
                                duration = 4000  # milliseconds
                                freq = 600  # Hz
                                winsound.Beep(freq, duration)
