# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:13:46 2019

@author: emrecemaksu
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


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

degerler = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=1)

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='softplus'))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adamax')
history = model.fit(x_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)
result = model.predict(x_test)
print(result)
print(y_test)
print(r2_score(y_test, result))
