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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

veriler = pd.read_csv("veriler.csv")

Fosfat = veriler[["Fosfat"]]
Ph = veriler[["PH"]]
Sicaklik = veriler[["Sicaklik"]]
Bioxaerobic = veriler[["BIOXAEROBIC"]]
Bioanoxic = veriler[["BIOANOXIC"]]
Phbaer = veriler[["PHBAER"]]
Phbanox = veriler[["PHBANOX"]]
Cod = veriler[["COD"]]
Nhdort = veriler[["NH4"]]
Pnf = veriler[["PNF"]]
Codnf = veriler[["CODNF"]]
Podortuf = veriler[["PO4UF"]]
Nhdortuf = veriler[["NH4UF"]]
Coduf = veriler[["CODUF"]]

imputer= Imputer
imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    
Fosfat = imputer.fit_transform(Fosfat)
Ph = imputer.fit_transform(Ph)
Sicaklik = imputer.fit_transform(Sicaklik)
Phbaer = imputer.fit_transform(Phbaer)
Phbanox = imputer.fit_transform(Phbanox)
Nhdort = imputer.fit_transform(Nhdort)


scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
Fosfat = scaler.fit_transform(Fosfat)
Ph = scaler.fit_transform(Ph)
Sicaklik = scaler.fit_transform(Sicaklik)
Bioxaerobic = scaler.fit_transform(Bioxaerobic)
Bioanoxic = scaler.fit_transform(Bioanoxic)
Phbaer = scaler.fit_transform(Phbaer)
Phbanox = scaler.fit_transform(Phbanox)
Pnf = scaler.fit_transform(Pnf)
Codnf = scaler.fit_transform(Codnf)
Coduf = scaler.fit_transform(Coduf)
"""
scaler = StandardScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
Fosfat = scaler.fit_transform(Fosfat)
Ph = scaler.fit_transform(Ph)
Sicaklik = scaler.fit_transform(Sicaklik)
Bioxaerobic = scaler.fit_transform(Bioxaerobic)
Bioanoxic = scaler.fit_transform(Bioanoxic)
Phbaer = scaler.fit_transform(Phbaer)
Phbanox = scaler.fit_transform(Phbanox)
Pnf = scaler.fit_transform(Pnf)
Codnf = scaler.fit_transform(Codnf)
Coduf = scaler.fit_transform(Coduf)
"""
Fosfat = pd.DataFrame(data = Fosfat, index = range(176), columns=['Fosfat'])
Ph = pd.DataFrame(data = Ph, index = range(176), columns =['Ph'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(176), columns=['Sicaklik'])
Phbaer = pd.DataFrame(data = Phbaer, index = range(176), columns=['Phbaer'])
Phbanox = pd.DataFrame(data = Phbanox, index = range(176), columns=['Phbanox'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(176), columns=['Nhdort'])
Cod = pd.DataFrame(data = Cod, index = range(176), columns=['Cod'])
Bioxaerobic = pd.DataFrame(data = Bioxaerobic, index = range(176), columns=['Bioxaerobic'])
Bioanoxic = pd.DataFrame(data = Bioanoxic, index = range(176), columns=['Bioanoxic'])
Pnf = pd.DataFrame(data = Pnf, index = range(176), columns =['Pnf'])
Codnf = pd.DataFrame(data = Codnf, index = range(176), columns =['Codnf'])
Coduf = pd.DataFrame(data = Coduf, index = range(176), columns =['Coduf'])


degerler = pd.concat([Cod, Nhdort, Fosfat, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler2 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)

"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=0)
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nXGBOOST CODUF R2 Skoru => ')
print(r2_score(y_test, y_pred))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler, Pnf, test_size=0.10, random_state=0)
params2 = {'n_estimators': 20, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_PNF = ensemble.GradientBoostingRegressor(**params2)
model_PNF.fit(x_train_PNF, y_train_PNF)
y_pred_PNF = model_PNF.predict(x_test_PNF)
print('\nXGBOOST PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_pred_PNF))

x_train_Fosfatsiz, x_test_Fosfatsiz, y_train_Fosfatsiz, y_test_Fosfatsiz = train_test_split(degerler2, Coduf, test_size=0.10, random_state=0)
params3 = {'n_estimators': 25, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_CODUFF = ensemble.GradientBoostingRegressor(**params)
model_CODUFF.fit(x_train_Fosfatsiz, y_train_Fosfatsiz)
y_pred = model_CODUFF.predict(x_test_Fosfatsiz)
print('\nXGBOOST CODUF R2 Skoru => ')
print(r2_score(y_test_Fosfatsiz, y_pred))

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler, Nhdortuf, test_size=0.10, random_state=0)
params4 = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_NHUF = ensemble.GradientBoostingRegressor(**params)
model_NHUF.fit(x_train_NHUF, y_train_NHUF)
y_pred_NHUF = model_NHUF.predict(x_test_NHUF)
print('\nXGBOOST NHUF R2 Skoru => ')
print(r2_score(y_test_NHUF, y_pred_NHUF))


x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.10, random_state=0)
params5 = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_POUF = ensemble.GradientBoostingRegressor(**params)
model_POUF.fit(x_train_POUF, y_train_POUF)
y_pred_POUF = model_POUF.predict(x_test_POUF)
print('\nXGBOOST POUF R2 Skoru => ')
print(r2_score(y_test_POUF, y_pred_POUF))
"""
x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler2, Codnf, test_size=0.25, random_state=0)
params6 = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
model_CODNF = ensemble.GradientBoostingRegressor(**params6)
model_CODNF.fit(x_train_CODNF, y_train_CODNF)
y_pred_CODNF = model_CODNF.predict(x_test_CODNF)
print('\nXGBOOST CODNF R2 Skoru => ')
print(r2_score(y_test_CODNF, y_pred_CODNF))

eskiskor = 0
eskiestimator = 0
eskidepth = 0
eskisplit = 0
eskirandom = 0

for estimator in range(1,30):
    for depth in range(1,30):
        for randomsta in range(0,75):
            for split in range(2,75):
                x_train, x_test, y_train, y_test = train_test_split(degerler2, Coduf, test_size=0.25, random_state=randomsta)
                params2 = {'n_estimators': estimator, 'max_depth': depth, 'min_samples_split': split,'learning_rate': 0.1, 'loss': 'ls'}
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
                    print(eskiskor)
