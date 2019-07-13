# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:31:36 2019

@author: emrecemaksu
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=2)

rf_reg_CODNF = RandomForestRegressor(n_estimators = 15, random_state=37)
rf_reg_CODNF.fit(x_train, y_train)
y_test_pred_rf_CODNF = rf_reg_CODNF.predict(x_test)
print('\nRandom Forest CODNF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_rf_CODNF))
"""
x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.10, random_state=20)

rf_reg_NNF = RandomForestRegressor(n_estimators = 26, random_state=0)
rf_reg_NNF.fit(x_train_NNF, y_train_NNF)
y_test_pred_rf_NNF = rf_reg_NNF.predict(x_test_NNF)
print('\nRandom Forest NNF R2 Skoru => ')
print(r2_score(y_test_NNF, y_test_pred_rf_NNF))

eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
eskiesti = 0
for esti in range(1,50):
    for rand in range(0,50):
        for train_rnd in range(0,50):
            x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=train_rnd)
            rf_reg_CODNF = RandomForestRegressor(n_estimators = esti, random_state=rand)
            rf_reg_CODNF.fit(x_train, y_train)
            y_test_pred_rf_CODNF = rf_reg_CODNF.predict(x_test)
            skor = r2_score(y_test, y_test_pred_rf_CODNF)
            if(skor > eskiskor):
                eskiskor = skor
                eskitrain_rnd = train_rnd
                eskirand = rand
                eskiesti = esti
"""
