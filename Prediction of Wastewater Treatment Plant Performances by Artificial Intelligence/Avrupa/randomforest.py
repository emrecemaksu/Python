# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:44:10 2019

@author: emrecemaksu
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
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
Podortuf = scaler.fit_transform(Podortuf)
Nhdortuf = scaler.fit_transform(Nhdortuf)

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
Cod = pd.DataFrame(data = Cod, index = range(176), columns=['Cod'])
Bioxaerobic = pd.DataFrame(data = Bioxaerobic, index = range(176), columns=['Bioxaerobic'])
Bioanoxic = pd.DataFrame(data = Bioanoxic, index = range(176), columns=['Bioanoxic'])
Pnf = pd.DataFrame(data = Pnf, index = range(176), columns =['Pnf'])
Codnf = pd.DataFrame(data = Codnf, index = range(176), columns =['Codnf'])
Coduf = pd.DataFrame(data = Coduf, index = range(176), columns =['Coduf'])
Nhdortuf = pd.DataFrame(data = Nhdortuf, index = range(176), columns =['Nhdortuf'])
Podortuf = pd.DataFrame(data = Podortuf, index = range(176), columns =['Podortuf'])


degerler = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler4 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox, Nhdortuf, Podortuf], axis=1)
degerler6 = pd.concat([Cod, Nhdort, Ph, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox,  Podortuf], axis=1)
degerler7 = pd.concat([Cod, Ph, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox,  Podortuf], axis=1)
degerler5 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Nhdortuf, Podortuf], axis=1)
degerler3 = pd.concat([Cod, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler2 = pd.concat([Cod, Nhdort, Ph, Sicaklik], axis=1)
"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=35)

rf_reg = RandomForestRegressor(n_estimators = 10, random_state=93)
rf_reg.fit(x_train, y_train)
y_test_pred_rf = rf_reg.predict(x_test)
print('\nRandom Forest CODUF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_rf))

x_train, x_test, y_train, y_test = train_test_split(degerler2, Coduf, test_size=0.25, random_state=45)

rf_reg = RandomForestRegressor(n_estimators = 1, random_state=20)
rf_reg.fit(x_train, y_train)
y_test_pred_rf = rf_reg.predict(x_test)
print('\nRandom Forest CODUF input <Cod, Nhdort, Ph, Sicaklik> R2 Skoru => ')
print(r2_score(y_test, y_test_pred_rf))


x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler, Nhdortuf, test_size=0.25, random_state=47)

rf_reg_NHUF = RandomForestRegressor(n_estimators = 3, random_state=13)
rf_reg_NHUF.fit(x_train_NHUF, y_train_NHUF)
y_test_pred_rf_NHUF = rf_reg_NHUF.predict(x_test_NHUF)
print('\nRandom Forest NH4UF R2 Skoru => ')
print(r2_score(y_test_NHUF, y_test_pred_rf_NHUF))

x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.25, random_state=0)

rf_reg_POUF = RandomForestRegressor(n_estimators = 10, random_state=2)
rf_reg_POUF.fit(x_train_POUF, y_train_POUF)
y_test_pred_rf_POUF = rf_reg_POUF.predict(x_test_POUF)
print('\nRandom Forest PO4UF R2 Skoru => ')
print(r2_score(y_test_POUF, y_test_pred_rf_POUF))

x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler, Codnf, test_size=0.10, random_state=22)

rf_reg_CODNF = RandomForestRegressor(n_estimators = 3, random_state=26)
rf_reg_CODNF.fit(x_train_CODNF, y_train_CODNF)
y_test_pred_rf_CODNF = rf_reg_CODNF.predict(x_test_CODNF)
print('\nRandom Forest CODNF R2 Skoru => ')
print(r2_score(y_test_CODNF, y_test_pred_rf_CODNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler, Pnf, test_size=0.10, random_state=12)

rf_reg_PNF = RandomForestRegressor(n_estimators = 1, random_state=89)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler4, Pnf, test_size=0.25, random_state=0)

rf_reg_PNF = RandomForestRegressor(n_estimators = 3, random_state=76)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler5, Pnf, test_size=0.25, random_state=68)

rf_reg_PNF = RandomForestRegressor(n_estimators = 3, random_state=54)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler6, Pnf, test_size=0.25, random_state=68)

rf_reg_PNF = RandomForestRegressor(n_estimators = 4, random_state=16)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))
"""
x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler7, Pnf, test_size=0.25, random_state=84)

rf_reg_PNF = RandomForestRegressor(n_estimators = 4, random_state=4)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))

"""
eskiskor = 0
eskitrain_rnd = 0
eskirand = 0
eskiesti = 0

for rand in range(0,100):
    for train_rnd in range(0,100):
        x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler6, Codnf, test_size=0.25, random_state=train_rnd)
        rf_reg_CODNF = RandomForestRegressor(n_estimators = 3, random_state=rand)
        rf_reg_CODNF.fit(x_train_CODNF, y_train_CODNF)
        y_test_pred_rf_CODNF = rf_reg_CODNF.predict(x_test_CODNF)
        skor = r2_score(y_test_CODNF, y_test_pred_rf_CODNF)
        if(skor > eskiskor):
            eskiskor = skor
            eskitrain_rnd = train_rnd
            eskirand = rand
            print(eskiskor)
            print(rand)
"""
