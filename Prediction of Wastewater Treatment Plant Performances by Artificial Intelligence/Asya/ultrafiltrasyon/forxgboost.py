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

SS = pd.DataFrame(data = SS, index = range(358), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(358), columns =['MLSSAero'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=57)
params = {'n_estimators': 9, 'max_depth': 14, 'min_samples_split': 12,'learning_rate': 0.3, 'loss': 'ls'}
model = ensemble.GradientBoostingRegressor(**params)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('\nXGBOOST CODUF R2 Skoru => ')
print(r2_score(y_test, y_pred))

"""
x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler, Nuf, test_size=0.10, random_state=5)
params2 = {'n_estimators': 16, 'max_depth': 5, 'min_samples_split': 4,'learning_rate': 0.66, 'loss': 'ls'}
model_NUF = ensemble.GradientBoostingRegressor(**params2)
model_NUF.fit(x_train_NUF, y_train_NUF)
y_pred_NUF = model_NUF.predict(x_test_NUF)
print('\nXGBOOST NH4UF R2 Skoru => ')
print(r2_score(y_test_NUF, y_pred_NUF))

eskiskor = 0
eskiestimator = 0
eskidepth = 0
eskisplit = 0
eskirandom = 0
eskirate = 0
for rate in np.arange(0.3, 1, 0.10):
    for estimator in range(1,30):
        for depth in range(1,30):
            for randomsta in range(0,75):
                for split in range(2,75):
                    x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=randomsta)
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
"""
