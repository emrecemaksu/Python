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

veriler = pd.read_csv("veriler_nf.csv")
Nhdort = veriler[["NH4N"]]
ALF = veriler[["Averageleachateflow"]]
SS = veriler[["SS"]]
Sicaklik = veriler[["Temperature"]]
Cod = veriler[["COD"]]
MLSSAero = veriler[["MLSSaerobic"]]
Nnf = veriler[["NNF"]]
Codnf = veriler[["CODNF"]]
print(veriler)
imputer= Imputer
imputer= Imputer(missing_values='NaN', strategy = 'mean', axis=0 )    
SS = imputer.fit_transform(SS)
MLSSAero = imputer.fit_transform(MLSSAero)

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])
degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.10, random_state=20)
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=6, activation='softplus'))
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
print(prediction)
print(y_test)
print(r2_score(y_test, prediction))

"""
x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.10, random_state=20)
# define base model
def baseline_model_NNF():
	# create model
	model = Sequential()
	model.add(Dense(12, input_dim=6, activation='softplus'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adamax')
	return model

# fix random seed for reproducibility
seed = 20
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator_NNF = KerasRegressor(build_fn=baseline_model_NNF, epochs=200, batch_size=4, verbose=0)
KFold(n_splits=6, random_state=seed)
estimator_NNF.fit(x_train_NNF, y_train_NNF)
prediction_NNF = estimator_NNF.predict(x_test_NNF)
print(prediction_NNF)
print(y_test_NNF)
print(r2_score(y_test_NNF, prediction_NNF))
"""
