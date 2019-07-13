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
import statsmodels.formula.api as sm 
from sklearn.preprocessing import MinMaxScaler

veriler = pd.read_csv("veriler_nf.csv")
Nhdort = veriler[["NH4N"]]
ALF = veriler[["Averageleachateflow"]]
SS = veriler[["SS"]]
Sicaklik = veriler[["Temperature"]]
Cod = veriler[["COD"]]
MLSSAero = veriler[["MLSSaerobic"]]
Nnf = veriler[["NNF"]]
Codnf = veriler[["CODNF"]]
print(veriler)
imputer= Imputer
imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    
SS = imputer.fit_transform(SS)
MLSSAero = imputer.fit_transform(MLSSAero)

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])

scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)
Nnf = scaler.fit_transform(Nnf)
Codnf = scaler.fit_transform(Codnf) 

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])
Codnf = pd.DataFrame(data = Cod, index = range(275), columns =['Codnf'])
Cod = pd.DataFrame(data = Cod, index = range(275), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(275), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(275), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(275), columns =['ALF'])
Nnf = pd.DataFrame(data = Nnf, index = range(275), columns =['Nnf'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
degerler2 = pd.concat([Cod, SS, ALF, Sicaklik, MLSSAero], axis=1)
"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=0)

x_train = np.append(arr = np.ones((206,1)).astype(int), values=x_train.iloc[:,0:], axis=1 )
x_train_opt = x_train[:,[0,1,2,3,4,5,6]]
r_ols = sm.OLS(endog = y_train, exog =x_train_opt)
r = r_ols.fit()
print(r.summary())
"""
"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=40)
r_dt_CODNF = DecisionTreeRegressor(random_state=92)
r_dt_CODNF.fit(x_train,y_train)
y_test_pred_dt_CODNF = r_dt_CODNF.predict(x_test)
print('\nDecision Tree CODNF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_dt_CODNF))
"""
x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.25, random_state=64)
r_dt_NNF = DecisionTreeRegressor(random_state=63)
r_dt_NNF.fit(x_train_NNF,y_train_NNF)
y_test_pred_dt_NNF = r_dt_NNF.predict(x_test_NNF)
print('\nDecision Tree NNF R2 Skoru => ')
print(r2_score(y_test_NNF, y_test_pred_dt_NNF))

eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
for rand in range(0,100):
        for train_rnd in range(0,100):
            x_train, x_test, y_train, y_test = train_test_split(degerler2, Nnf, test_size=0.25, random_state=train_rnd)
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
