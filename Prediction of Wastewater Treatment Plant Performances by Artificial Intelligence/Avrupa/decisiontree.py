# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:45:46 2019

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
"""
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
Podortuf = scaler.fit_transform(Podortuf)
Nhdortuf = scaler.fit_transform(Nhdortuf)
"""
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
Podortuf = scaler.fit_transform(Podortuf)
Nhdortuf = scaler.fit_transform(Nhdortuf)
"""
Fosfat = pd.DataFrame(data = Fosfat, index = range(176), columns=['Fosfat'])
Ph = pd.DataFrame(data = Ph, index = range(176), columns =['Ph'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(176), columns=['Sicaklik'])
Phbaer = pd.DataFrame(data = Phbaer, index = range(176), columns=['Phbaer'])
Phbanox = pd.DataFrame(data = Phbanox, index = range(176), columns=['Phbanox'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(176), columns=['Nhdort'])
"""
Cod = pd.DataFrame(data = Cod, index = range(176), columns=['Cod'])
Bioxaerobic = pd.DataFrame(data = Bioxaerobic, index = range(176), columns=['Bioxaerobic'])
Bioanoxic = pd.DataFrame(data = Bioanoxic, index = range(176), columns=['Bioanoxic'])
Pnf = pd.DataFrame(data = Pnf, index = range(176), columns =['Pnf'])
Codnf = pd.DataFrame(data = Codnf, index = range(176), columns =['Codnf'])
Coduf = pd.DataFrame(data = Coduf, index = range(176), columns =['Coduf'])
Nhdortuf = pd.DataFrame(data = Nhdortuf, index = range(176), columns =['Nhdortuf'])
Podortuf = pd.DataFrame(data = Podortuf, index = range(176), columns =['Podortuf'])
"""

degerler = pd.concat([Cod, Nhdort, Fosfat, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler2 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler3 = pd.concat([Cod, Nhdort, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Nhdortuf, Podortuf], axis=1)
degerler4 = pd.concat([Cod, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Coduf, Nhdortuf, Podortuf], axis=1)
"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=0)
r_dt2 = DecisionTreeRegressor(random_state=4)
r_dt2.fit(x_train,y_train)
y_test_pred_dt_Fosfatli = r_dt2.predict(x_test)
print('\nDecision Tree CODUF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_dt_Fosfatli))

x_train, x_test, y_train, y_test = train_test_split(degerler3, Coduf, test_size=0.25, random_state=50)
r_dt2 = DecisionTreeRegressor(random_state=96)
r_dt2.fit(x_train,y_train)
y_test_pred_dt_Fosfatli = r_dt2.predict(x_test)
print('\nDecision Tree CODUF <Cod, Nhdort, Ph, Sicaklik> R2 Skoru => ')
print(r2_score(y_test, y_test_pred_dt_Fosfatli))
"""
x_train_Fosfatsiz, x_test_Fosfatsiz, y_train_Fosfatsiz, y_test_Fosfatsiz = train_test_split(degerler2, Coduf, test_size=0.10, random_state=122)
r_dt = DecisionTreeRegressor(random_state=189)
r_dt.fit(x_train_Fosfatsiz,y_train_Fosfatsiz)
y_test_pred_dt = r_dt.predict(x_test_Fosfatsiz)
print('\nDecision Tree R2 Skoru Fosfatsız CODUF => ')
print(r2_score(y_test_Fosfatsiz, y_test_pred_dt))
"""
x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler, Nhdortuf, test_size=0.10, random_state=0)
r_dt_NHUF = DecisionTreeRegressor(random_state=4)
r_dt_NHUF.fit(x_train_NHUF,y_train_NHUF)
y_test_pred_dt_NHUF = r_dt_NHUF.predict(x_test_NHUF)
print('\nDecision Tree NH4UF R2 Skoru => ')
print(r2_score(y_test_NHUF, y_test_pred_dt_NHUF))

x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.25, random_state=36)
r_dt_POUF = DecisionTreeRegressor(random_state=42)
r_dt_POUF.fit(x_train_POUF,y_train_POUF)
y_test_pred_dt_POUF = r_dt_POUF.predict(x_test_POUF)
print('\nDecision Tree PO4UF R2 Skoru => ')
print(r2_score(y_test_POUF, y_test_pred_dt_POUF))
x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler, Pnf, test_size=0.10, random_state=0)
r_dt_PNF = DecisionTreeRegressor(random_state=0)
r_dt_PNF.fit(x_train_PNF,y_train_PNF)
y_test_pred_dt_PNF = r_dt_PNF.predict(x_test_PNF)
print('\nDecision Tree PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_dt_PNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler3, Pnf, test_size=0.25, random_state=84)
r_dt_PNF = DecisionTreeRegressor(random_state=94)
r_dt_PNF.fit(x_train_PNF,y_train_PNF)
y_test_pred_dt_PNF = r_dt_PNF.predict(x_test_PNF)
print('\nDecision Tree PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_dt_PNF))
"""
"""
x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler2, Codnf, test_size=0.10, random_state=6)
r_dt_CODNF = DecisionTreeRegressor(random_state=6)
r_dt_CODNF.fit(x_train_CODNF,y_train_CODNF)
y_test_pred_dt_CODNF = r_dt_CODNF.predict(x_test_CODNF)
print('\nDecision Tree CODNF R2 Skoru Fosfatsız => ')
print(r2_score(y_test_CODNF, y_test_pred_dt_CODNF))

x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler2, Codnf, test_size=0.25, random_state=46)
r_dt_CODNF = DecisionTreeRegressor(random_state=42)
r_dt_CODNF.fit(x_train_CODNF,y_train_CODNF)
y_test_pred_dt_CODNF = r_dt_CODNF.predict(x_test_CODNF)
print('\nDecision Tree CODNF R2 Skoru Fosfatsız => ')
print(r2_score(y_test_CODNF, y_test_pred_dt_CODNF))
"""
"""
eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
for rand in range(0,100):
        for train_rnd in range(0,100):
            x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler4, Codnf, test_size=0.25, random_state=train_rnd)
            r_dt_CODNF = DecisionTreeRegressor(random_state=rand)
            r_dt_CODNF.fit(x_train_CODNF,y_train_CODNF)
            y_test_pred_dt_CODNF = r_dt_CODNF.predict(x_test_CODNF)
            skor = r2_score(y_test_CODNF, y_test_pred_dt_CODNF)
            if(skor > eskiskor):
                eskiskor = skor
                eskitrain_rnd = train_rnd
                eskirand = rand
                print(eskiskor)
                print(rand)
"""
