# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 10:47:19 2019

@author: emrecemaksu
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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

Fosfat = pd.DataFrame(data = Fosfat, index = range(176), columns=['Fosfat'])
Ph = pd.DataFrame(data = Ph, index = range(176), columns =['Ph'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(176), columns=['Sicaklik'])
Phbaer = pd.DataFrame(data = Phbaer, index = range(176), columns=['Phbaer'])
Phbanox = pd.DataFrame(data = Phbanox, index = range(176), columns=['Phbanox'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(176), columns=['Nhdort'])
degerler = pd.concat([Cod, Nhdort, Fosfat, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=0)
svr_reg = SVR(kernel = 'rbf')
sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x_train)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y_train)
sc3 = StandardScaler()
x_test_olcekli = sc3.fit_transform(x_test)
sc4 = StandardScaler()
y_test_olcekli = sc4.fit_transform(y_test)
svr_reg.fit(x_olcekli,y_olcekli)
y_test_pred_SVR = svr_reg.predict(x_test_olcekli)
print('\nSVR R2 Skoru => ')
print(r2_score(y_test_olcekli, y_test_pred_SVR))

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler, Nhdortuf, test_size=0.10, random_state=0)
svr_reg_NHUF = SVR(kernel = 'rbf')
sc1_NHUF = StandardScaler()
x_olcekli_NHUF = sc1_NHUF.fit_transform(x_train_NHUF)
sc2_NHUF = StandardScaler()
y_olcekli_NHUF = sc2_NHUF.fit_transform(y_train_NHUF)
sc3_NHUF = StandardScaler()
x_test_olcekli_NHUF = sc3_NHUF.fit_transform(x_test_NHUF)
sc4_NHUF = StandardScaler()
y_test_olcekli_NHUF = sc4_NHUF.fit_transform(y_test_NHUF)
svr_reg_NHUF.fit(x_olcekli_NHUF,y_olcekli_NHUF)
y_test_pred_SVR_NHUF = svr_reg_NHUF.predict(x_test_olcekli_NHUF)
print('\nSVR NH4UF R2 Skoru => ')
print(r2_score(y_test_olcekli_NHUF, y_test_pred_SVR_NHUF))

x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.10, random_state=0)
svr_reg_POUF = SVR(kernel = 'rbf')
sc1_POUF = StandardScaler()
x_olcekli_POUF = sc1_POUF.fit_transform(x_train_POUF)
sc2_POUF = StandardScaler()
y_olcekli_POUF = sc2_POUF.fit_transform(y_train_POUF)
sc3_POUF = StandardScaler()
x_test_olcekli_POUF = sc3_POUF.fit_transform(x_test_POUF)
sc4_POUF = StandardScaler()
y_test_olcekli_POUF = sc4_POUF.fit_transform(y_test_POUF)
svr_reg_POUF.fit(x_olcekli_POUF,y_olcekli_POUF)
y_test_pred_SVR_POUF = svr_reg_POUF.predict(x_test_olcekli_POUF)
print('\nSVR PO4UF R2 Skoru => ')
print(r2_score(y_test_olcekli_POUF, y_test_pred_SVR_POUF))

x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler, Codnf, test_size=0.10, random_state=0)
svr_reg_CODNF = SVR(kernel = 'rbf')
sc1_CODNF = StandardScaler()
x_olcekli_CODNF = sc1_CODNF.fit_transform(x_train_CODNF)
sc2_CODNF = StandardScaler()
y_olcekli_CODNF = sc2_CODNF.fit_transform(y_train_CODNF)
sc3_CODNF = StandardScaler()
x_test_olcekli_CODNF = sc3_CODNF.fit_transform(x_test_CODNF)
sc4_CODNF = StandardScaler()
y_test_olcekli_CODNF = sc4_CODNF.fit_transform(y_test_CODNF)
svr_reg_CODNF.fit(x_olcekli_CODNF,y_olcekli_CODNF)
y_test_pred_SVR_CODNF = svr_reg_CODNF.predict(x_test_olcekli_CODNF)
print('\nSVR CODNF R2 Skoru => ')
print(r2_score(y_test_olcekli_CODNF, y_test_pred_SVR_CODNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler, Pnf, test_size=0.10, random_state=0)
svr_reg_PNF = SVR(kernel = 'rbf')
sc1_PNF = StandardScaler()
x_olcekli_PNF = sc1_PNF.fit_transform(x_train_PNF)
sc2_PNF = StandardScaler()
y_olcekli_PNF = sc2_PNF.fit_transform(y_train_PNF)
sc3_PNF = StandardScaler()
x_test_olcekli_PNF = sc3_PNF.fit_transform(x_test_PNF)
sc4_PNF = StandardScaler()
y_test_olcekli_PNF = sc4_PNF.fit_transform(y_test_PNF)
svr_reg_PNF.fit(x_olcekli_PNF,y_olcekli_PNF)
y_test_pred_SVR_PNF = svr_reg_PNF.predict(x_test_olcekli_PNF)
print('\nSVR PNF R2 Skoru => ')
print(r2_score(y_test_olcekli_PNF, y_test_pred_SVR_PNF))
