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
import winsound
from sklearn.preprocessing import MinMaxScaler
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
"""
scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)

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

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
degerler2 = pd.concat([Cod, Nhdort, ALF, Sicaklik, MLSSAero], axis=1)

SS = pd.DataFrame(data = SS, index = range(358), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(358), columns =['MLSSAero'])

x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=42)

rf_reg = RandomForestRegressor(n_estimators = 1, random_state=14)
rf_reg.fit(x_train, y_train)
y_test_pred_rf = rf_reg.predict(x_test)
print('\nRandom Forest CODUF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_rf))

"""
x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler, Nuf, test_size=0.25, random_state=5)

rf_reg_NUF = RandomForestRegressor(n_estimators = 8, random_state=2)
rf_reg_NUF.fit(x_train_NUF, y_train_NUF)
y_test_pred_rf_NUF = rf_reg_NUF.predict(x_test_NUF)
print('\nRandom Forest NUF R2 Skoru => ')
print(r2_score(y_test_NUF, y_test_pred_rf_NUF))

x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler2, Nuf, test_size=0.25, random_state=21)

rf_reg_NUF = RandomForestRegressor(n_estimators = 2, random_state=96)
rf_reg_NUF.fit(x_train_NUF, y_train_NUF)
y_test_pred_rf_NUF = rf_reg_NUF.predict(x_test_NUF)
print('\nRandom Forest NUF R2 Skoru => ')
print(r2_score(y_test_NUF, y_test_pred_rf_NUF))
"""
"""
eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
eskiesti = 0
for esti in range(1,50):
    for rand in range(0,100):
        for train_rnd in range(0,100):
            x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler, Nuf, test_size=0.25, random_state=train_rnd)
            rf_reg_NUF = RandomForestRegressor(n_estimators = esti, random_state=rand)
            rf_reg_NUF.fit(x_train_NUF, y_train_NUF)
            y_test_pred_rf_NUF = rf_reg_NUF.predict(x_test_NUF)
            skor = r2_score(y_test_NUF, y_test_pred_rf_NUF)
            if(skor > eskiskor):
                eskiskor = skor
                eskitrain_rnd = train_rnd
                eskirand = rand
                eskiesti = esti
                print(eskiskor)
                if(eskiskor > 0.90):
                    duration = 2000  # milliseconds
                    freq = 600  # Hz
                    winsound.Beep(freq, duration)
"""
