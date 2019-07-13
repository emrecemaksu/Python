# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 21:13:46 2019

@author: emrecemaksu
"""

import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
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

degerler = pd.concat([Cod, Nhdort, Fosfat, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler3 = pd.concat([Cod, Nhdort], axis=1)

def baseline_model_NHUF():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler3, Nhdortuf, test_size=0.10, random_state=1)
# fix random seed for reproducibility

eskiskor = 0
eskiepoc = 0
eskiseed = 0
eskisplit = 0
for epoc in range(1,20):
    for size in range(1,20):
        for split in range(2,20):
            seed = 20
            numpy.random.seed(seed)
            # evaluate model with standardized dataset
            estimator = KerasRegressor(build_fn=baseline_model_NHUF, epochs=epoc, batch_size=size, verbose=0)
            KFold(n_splits=split, random_state=seed)
            estimator.fit(x_train_NHUF, y_train_NHUF)
            prediction_NHUF = estimator.predict(x_test_NHUF)
            skor = r2_score(y_test_NHUF, prediction_NHUF)
            if(skor > eskiskor):
                eskiskor = skor
                eskisplit = split
                eskiseed = seed
                eskiepoc = epoc
""""
x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=2)

# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=9, activation='softplus'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=4, verbose=0)
KFold(n_splits=6, random_state=seed)
estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print(prediction),
print(y_test)
print(r2_score(y_test, prediction))

def baseline_model_NHUF():
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler3, Nhdortuf, test_size=0.10, random_state=1)
# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model_NHUF, epochs=150, batch_size=4, verbose=0)
KFold(n_splits=3, random_state=seed)
estimator.fit(x_train_NHUF, y_train_NHUF)
prediction_NHUF = estimator.predict(x_test_NHUF)
print(prediction_NHUF)
print(y_test_NHUF)
print(r2_score(y_test_NHUF, prediction_NHUF))

x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.10, random_state=2)

# define base model
def baseline_model_POUF():
	# create model
	model = Sequential()
	model.add(Dense(9, input_dim=9, activation='softplus'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator_POUF = KerasRegressor(build_fn=baseline_model_POUF, epochs=50, batch_size=1, verbose=0)
KFold(n_splits=2, random_state=seed)
estimator_POUF.fit(x_train_POUF, y_train_POUF)
prediction_POUF = estimator_POUF.predict(x_test_POUF)
print(prediction_POUF)
print(y_test_POUF)
print(r2_score(y_test_POUF, prediction_POUF))

x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler, Codnf, test_size=0.10, random_state=2)

# define base model
def baseline_model_CODNF():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=9, activation='softplus'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator_CODNF = KerasRegressor(build_fn=baseline_model_CODNF, epochs=200, batch_size=4, verbose=0)
KFold(n_splits=6, random_state=seed)
estimator_CODNF.fit(x_train_CODNF, y_train_CODNF)
prediction_CODNF = estimator_CODNF.predict(x_test_CODNF)
print(prediction_CODNF)
print(y_test_CODNF)
print(r2_score(y_test_CODNF, prediction_CODNF))

x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerler, Pnf, test_size=0.10, random_state=1)

# define base model
def baseline_model_PNF():
	# create model
	model = Sequential()
	model.add(Dense(9, input_dim=9, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator_PNF = KerasRegressor(build_fn=baseline_model_PNF, epochs=100, batch_size=1, verbose=0)
KFold(n_splits=2, random_state=seed)
estimator_PNF.fit(x_train_PNF, y_train_PNF)
prediction_PNF = estimator_PNF.predict(x_test_PNF)
print(prediction_PNF)
print(y_test_PNF)
print(r2_score(y_test_PNF, prediction_PNF))
"""
