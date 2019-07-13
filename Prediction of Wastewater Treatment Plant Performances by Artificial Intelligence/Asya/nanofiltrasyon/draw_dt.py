#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:48:15 2019

@author: emrecemaksu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

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

"""
SS = SS.values
plt.plot(SS)
plt.show()
"""
"""
ALF = ALF.values
print(ALF)
plt.plot(ALF)
plt.show()
"""

scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)
Nnf = scaler.fit_transform(Nnf)
Codnf = scaler.fit_transform(Codnf) 

SS = pd.DataFrame(data = SS, index = range(275), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(275), columns =['MLSSAero'])
Codnf = pd.DataFrame(data = Cod, index = range(275), columns =['Codnf'])
Cod = pd.DataFrame(data = Cod, index = range(275), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(275), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(275), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(275), columns =['ALF'])
Nnf = pd.DataFrame(data = Nnf, index = range(275), columns =['Nnf'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
degerler2 = pd.concat([Cod, SS, ALF, Sicaklik, MLSSAero], axis=1)


x_train_NNF, x_test_NNF, y_train_NNF, y_test_NNF = train_test_split(degerler, Nnf, test_size=0.25, random_state=64)
r_dt_NNF = DecisionTreeRegressor(random_state=63)
r_dt_NNF.fit(x_train_NNF,y_train_NNF)
y_test_pred_dt_NNF = r_dt_NNF.predict(x_test_NNF)
print('\nDecision Tree NNF R2 Skoru => ')
print(r2_score(y_test_NNF, y_test_pred_dt_NNF))

y_test_pred_dt_NNF = pd.DataFrame(data = y_test_pred_dt_NNF, index = range(69), columns =['y_test_pred_dt_NNF'])
#y_test_pred_dt_CODNF = y_test_pred_dt_CODNF.sort_index()
y_test_NNF.index = range(69)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_NNF, label='Real Value', color='red')
ax.plot(y_test_pred_dt_NNF, label='Predict', color='blue')
plt.title("Decision Tree N-NF Permeate R2 %94.5 (MinMax Scaler)")
plt.xlabel("GUN")
plt.ylabel("N-NF (mg/L)")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()
"""
x_train, x_test, y_train, y_test = train_test_split(degerler, Codnf, test_size=0.25, random_state=40)
r_dt_CODNF = DecisionTreeRegressor(random_state=92)
r_dt_CODNF.fit(x_train,y_train)
y_test_pred_dt_CODNF = r_dt_CODNF.predict(x_test)
print('\nDecision Tree CODNF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_dt_CODNF))


#x_test = x_test.sort_index()
#y_test = y_test.sort_index()
y_test_pred_dt_CODNF = pd.DataFrame(data = y_test_pred_dt_CODNF, index = range(69), columns =['y_test_pred_dt_CODNF'])
#y_test_pred_dt_CODNF = y_test_pred_dt_CODNF.sort_index()
y_test.index = range(69)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test, label='Real Value', color='red')
ax.plot(y_test_pred_dt_CODNF, label='Predict', color='blue')
plt.title("Decision Tree COD-NF Permeate R2 %99.99 (MinMax Scaler)")
plt.xlabel("GUN")
plt.ylabel("KOI-NF (mg/L)")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()

dot_data = StringIO()

export_graphviz(r_dt_CODNF, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('komurcuoda_dt_CODNF.png')
"""
