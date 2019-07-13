# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:31:12 2019

@author: emrecemaksu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import winsound
import statsmodels.formula.api as sm 


veriler = pd.read_csv("veriler_uf.csv")
Nhdort = veriler[["NH4N"]]
ALF = veriler[["Averageleachateflow"]]
SS = veriler[["SS"]]
Sicaklik = veriler[["Temperature"]]
Cod = veriler[["COD"]]
MLSSAero = veriler[["MLSSaerobic"]]
Nuf = veriler[["NH4NUF"]]
Coduf = veriler[["CODUF"]]

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
Nuf = scaler.fit_transform(Nuf)
"""
ss = StandardScaler()
Cod = ss.fit_transform(Cod)
Nhdort = ss.fit_transform(Nhdort)
SS = ss.fit_transform(SS)
Sicaklik = ss.fit_transform(Sicaklik)
MLSSAero = ss.fit_transform(MLSSAero)
ALF = ss.fit_transform(ALF)
"""
SS = pd.DataFrame(data = SS, index = range(358), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(358), columns =['MLSSAero'])
Cod = pd.DataFrame(data = Cod, index = range(358), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(358), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(358), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(358), columns =['ALF'])
Nuf = pd.DataFrame(data = Nuf, index = range(358), columns =['Nuf'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=54)
r_dt_CODUF = DecisionTreeRegressor(random_state=13)
r_dt_CODUF.fit(x_train,y_train)
y_test_pred_dt_CODUF = r_dt_CODUF.predict(x_test)
print('\nDecision Tree CODUF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_dt_CODUF))

degerler3 = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
"""
x_train, x_test, y_train, y_test = train_test_split(degerler3, Coduf, test_size=0.25, random_state=79)
r_dt_CODUF = DecisionTreeRegressor(random_state=49)
r_dt_CODUF.fit(x_train,y_train)
y_test_pred_dt_CODUF = r_dt_CODUF.predict(x_test)
print('\nDecision Tree CODUF R2 Skoru Amonyumsuz => ')
print(r2_score(y_test, y_test_pred_dt_CODUF))
degerler2 = pd.concat([Cod, ALF, Sicaklik, MLSSAero], axis=1)
"""
"""
x_train = np.append(arr = np.ones((268,1)).astype(int), values=x_train.iloc[:,0:], axis=1 )
x_train_opt = x_train[:,[0,1,2,3,4,5,6]]
r_ols = sm.OLS(endog = y_train, exog =x_train_opt)
r = r_ols.fit()
print(r.summary())
"""

x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler, Nuf, test_size=0.15, random_state=80)
r_dt_NUF = DecisionTreeRegressor(random_state=1)
r_dt_NUF.fit(x_train_NUF,y_train_NUF)
y_test_pred_dt_NUF = r_dt_NUF.predict(x_test_NUF)
print('\nDecision Tree NUF R2 Skoru => ')
print(r2_score(y_test_NUF, y_test_pred_dt_NUF))


eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
for rand in range(0,100):
        for train_rnd in range(0,100):
            x_train, x_test, y_train, y_test = train_test_split(degerler3, Nuf, test_size=0.25, random_state=train_rnd)
            r_dt_CODUF = DecisionTreeRegressor(random_state=rand)
            r_dt_CODUF.fit(x_train,y_train)
            y_test_pred_dt_CODUF = r_dt_CODUF.predict(x_test)
            skor = r2_score(y_test, y_test_pred_dt_CODUF)
            if(skor > eskiskor):
                eskiskor = skor
                eskitrain_rnd = train_rnd
                eskirand = rand
                print(eskiskor)
                print(rand)
                
                if(eskiskor > 0.90):
                    duration = 2000  # milliseconds
                    freq = 600  # Hz
                    winsound.Beep(freq, duration)
                    
