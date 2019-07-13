# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:30:49 2019

@author: emrecemaksu
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

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

x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.10, random_state=20)

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
print('\nSVR CODNF R2 Skoru => ')
print(r2_score(y_test_olcekli, y_test_pred_SVR))

x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.10, random_state=20)
svr_reg = SVR(kernel = 'rbf')
sc5 = StandardScaler()
x_olcekli = sc5.fit_transform(x_train_NNF)
sc6 = StandardScaler()
y_olcekli = sc6.fit_transform(y_train_NNF)
sc7 = StandardScaler()
x_test_olcekli = sc7.fit_transform(x_test_NNF)
sc8 = StandardScaler()
y_test_olcekli = sc8.fit_transform(y_test_NNF)
svr_reg.fit(x_olcekli,y_olcekli)
y_test_pred_SVR = svr_reg.predict(x_test_olcekli)
print('\nSVR NNF R2 Skoru => ')
print(r2_score(y_test_olcekli, y_test_pred_SVR))
