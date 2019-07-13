#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:47:40 2019

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
#import winsound
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
#import statsmodels.formula.api as sm 


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

scaler = MinMaxScaler()
Cod = scaler.fit_transform(Cod)
Nhdort = scaler.fit_transform(Nhdort)
SS = scaler.fit_transform(SS)
Sicaklik = scaler.fit_transform(Sicaklik)
MLSSAero = scaler.fit_transform(MLSSAero)
ALF = scaler.fit_transform(ALF)
Nuf = scaler.fit_transform(Nuf)

SS = pd.DataFrame(data = SS, index = range(358), columns=['SS'])
MLSSAero = pd.DataFrame(data = MLSSAero, index = range(358), columns =['MLSSAero'])
Cod = pd.DataFrame(data = Cod, index = range(358), columns =['Cod'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(358), columns =['Nhdort'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(358), columns =['Sicaklik'])
ALF = pd.DataFrame(data = ALF, index = range(358), columns =['ALF'])
Nuf = pd.DataFrame(data = Nuf, index = range(358), columns =['Nuf'])

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)

x_train_NUF, x_test_NUF, y_train_NUF, y_test_NUF = train_test_split(degerler, Nuf, test_size=0.15, random_state=80)
r_dt_NUF = DecisionTreeRegressor(random_state=1)
r_dt_NUF.fit(x_train_NUF,y_train_NUF)
y_test_pred_dt_NUF = r_dt_NUF.predict(x_test_NUF)
print('\nDecision Tree NUF R2 Skoru => ')
print(r2_score(y_test_NUF, y_test_pred_dt_NUF))

#x_test_NUF = x_test_NUF.sort_index()
#y_test_NUF = y_test_NUF.sort_index()
y_test_pred_dt_NUF = pd.DataFrame(data = y_test_pred_dt_NUF, index = range(54), columns =['y_test_pred_dt_POUF'])
y_test_pred_dt_NUF = y_test_pred_dt_NUF.sort_index()
y_test_NUF.index = range(54)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_NUF, label='Real Value', color='red')
ax.plot(y_test_pred_dt_NUF, label='Predict', color='blue')
plt.title("Decision Tree NH4-N UF Permeate R2 %83")
plt.xlabel("GUN")
plt.ylabel("NH4-N UF (mg-L)")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()

"""
dot_data = StringIO()

export_graphviz(r_dt_NUF, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('komurcuoda_dt_NUF.png')
"""
