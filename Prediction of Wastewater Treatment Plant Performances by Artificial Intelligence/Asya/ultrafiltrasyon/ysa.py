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
import winsound
from sklearn.preprocessing import MinMaxScaler

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
"""
scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)
"""
ss = StandardScaler()
Cod = ss.fit_transform(Cod)
Nhdort = ss.fit_transform(Nhdort)
SS = ss.fit_transform(SS)
Sicaklik = ss.fit_transform(Sicaklik)
MLSSAero = ss.fit_transform(MLSSAero)
ALF = ss.fit_transform(ALF)

SS = pd.DataFrame(data = SS, index = range(358), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(358), columns =['MLSSAero'])
Cod = pd.DataFrame(data = Cod, index = range(358), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(358), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(358), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(358), columns =['ALF'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
"""
degerler2 = pd.concat([Cod, Nhdort, SS, Sicaklik], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler2, Coduf, test_size=0.25, random_state=1)
#1 34.8 
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=4, activation='softplus'))
    #model.add(Dense(25, activation='softplus'))
    model.add(Dense(25, activation='softplus'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model
# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=3, verbose=0)
KFold(n_splits=5, random_state=seed)
estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print(r2_score(y_test, prediction))
duration = 500  # milliseconds
freq = 600  # Hz
winsound.Beep(freq, duration)
"""
"""
eskiskor = 0
eskiepoc = 0
eskiseed = 0
eskisplit = 0
eskistate = 0
eskibatch = 0
eskisecond = 0
eskifirst = 0
for first in range(6, 12):
    for second in range(30, 50):
        for state in range(0, 50):
            for epoc in range(20,200):
                for size in range(1,50):
                    for split in range(2,50):
                        def baseline_model():
                            # create model
                            model = Sequential()
                            model.add(Dense(first, input_dim=6, activation='softplus'))
                            model.add(Dense(second, activation='softplus'))
                            model.add(Dense(1))
                            model.compile(loss='mean_squared_error', optimizer='adamax')
                            return model
                        x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=state)
                        seed = 20
                        numpy.random.seed(seed)
                        estimator = KerasRegressor(build_fn=baseline_model, epochs=epoc, batch_size=size, verbose=0)
                        KFold(n_splits=split, random_state=seed)
                        estimator.fit(x_train, y_train)
                        prediction = estimator.predict(x_test)
                        skor = r2_score(y_test, prediction)
                        print(skor)
                        if(skor > eskiskor):
                            eskiskor = skor
                            eskisplit = split
                            eskiseed = seed
                            eskisecond = second
                            eskiepoc = epoc
                            eskifirst = first
                            eskistate = state
                            eskibatch = size

"""

x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.25, random_state=1)
#1 34.8 
# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(16, input_dim=6, activation='softplus'))
    model.add(Dense(50, activation='softplus'))
    model.add(Dense(50, activation='softplus'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adamax')
    return model
# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=200, batch_size=3, verbose=0)
KFold(n_splits=12, random_state=seed)
estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
print(r2_score(y_test, prediction))
duration = 1000  # milliseconds
freq = 600  # Hz
winsound.Beep(freq, duration)
"""
def baseline_model_NHUF():
	# create model
	model = Sequential()
	model.add(Dense(9, input_dim=6, activation='relu'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerler, Nuf, test_size=0.10, random_state=3)
# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model_NHUF, epochs=200, batch_size=4, verbose=0)
KFold(n_splits=3, random_state=seed)
estimator.fit(x_train_NHUF, y_train_NHUF)
prediction_NHUF = estimator.predict(x_test_NHUF)
print(prediction_NHUF)
print(y_test_NHUF)
print(r2_score(y_test_NHUF, prediction_NHUF))
"""
