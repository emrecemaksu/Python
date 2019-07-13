#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 12:28:41 2019

@author: pi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydot
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
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

"""
Fosfat = pd.DataFrame(data = Fosfat, index = range(176), columns=['Fosfat'])
Ph = pd.DataFrame(data = Ph, index = range(176), columns =['Ph'])
Sicaklik = pd.DataFrame(data = Sicaklik, index = range(176), columns=['Sicaklik'])
Phbaer = pd.DataFrame(data = Phbaer, index = range(176), columns=['Phbaer'])
Phbanox = pd.DataFrame(data = Phbanox, index = range(176), columns=['Phbanox'])
Nhdort = pd.DataFrame(data = Nhdort, index = range(176), columns=['Nhdort'])

degerlerx = pd.concat([Cod,Fosfat, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
"""
degerler = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler4 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox, Nhdortuf, Podortuf], axis=1)
degerler6 = pd.concat([Cod, Nhdort, Ph, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox,  Podortuf], axis=1)
degerler7 = pd.concat([Cod, Ph, Bioxaerobic, Bioanoxic, Phbaer, Coduf,Phbanox,  Podortuf], axis=1)
degerler5 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Nhdortuf, Podortuf], axis=1)
degerler3 = pd.concat([Cod, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler2 = pd.concat([Cod, Nhdort, Ph, Sicaklik], axis=1)
"""
"""
x_train_CODNF, x_test_CODNF, y_train_CODNF, y_test_CODNF = train_test_split(degerler, Codnf, test_size=0.10, random_state=22)

rf_reg_CODNF = RandomForestRegressor(n_estimators = 3, random_state=26)
rf_reg_CODNF.fit(x_train_CODNF, y_train_CODNF)
y_test_pred_rf_CODNF = rf_reg_CODNF.predict(x_test_CODNF)
tree = rf_reg_CODNF.estimators_[2]
print('\nRandom Forest CODNF R2 Skoru => ')
print(r2_score(y_test_CODNF, y_test_pred_rf_CODNF))
"""

x_train_NHUF, x_test_NHUF, y_train_NHUF, y_test_NHUF = train_test_split(degerlerx, Nhdortuf, test_size=0.25, random_state=47)

rf_reg_NHUF = RandomForestRegressor(n_estimators = 3, random_state=13)
rf_reg_NHUF.fit(x_train_NHUF, y_train_NHUF)
y_test_pred_rf_NHUF = rf_reg_NHUF.predict(x_test_NHUF)
tree = rf_reg_NHUF.estimators_[0]
print('\nRandom Forest NH4UF R2 Skoru => ')
print(r2_score(y_test_NHUF, y_test_pred_rf_NHUF))

y_test_pred_rf_NHUF = pd.DataFrame(data = y_test_pred_rf_NHUF, index = range(44), columns =['y_test_pred_rf_PNF'])
#y_test_pred_rf_PNF = y_test_pred_rf_PNF.sort_index()
y_test_NHUF.index = range(44)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_NHUF, label='Real Value', color='red')
ax.plot(y_test_pred_rf_NHUF, label='Predict', color='blue')
plt.title("Random Forest NH4UF R2 %86.6")
plt.xlabel("GUN")
plt.ylabel("NH4-N-UF mgP-L")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()

"""
x_train_PNF, x_test_PNF, y_train_PNF, y_test_PNF = train_test_split(degerlerx, Pnf, test_size=0.10, random_state=12)

rf_reg_PNF = RandomForestRegressor(n_estimators = 1, random_state=89)
rf_reg_PNF.fit(x_train_PNF, y_train_PNF)
y_test_pred_rf_PNF = rf_reg_PNF.predict(x_test_PNF)
print('\nRandom Forest PNF R2 Skoru => ')
print(r2_score(y_test_PNF, y_test_pred_rf_PNF))
tree = rf_reg_PNF.estimators_[0]


y_test_pred_rf_PNF = pd.DataFrame(data = y_test_pred_rf_PNF, index = range(18), columns =['y_test_pred_rf_PNF'])
#y_test_pred_rf_PNF = y_test_pred_rf_PNF.sort_index()
y_test_PNF.index = range(18)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_PNF, label='Real Value', color='red')
ax.plot(y_test_pred_rf_PNF, label='Predict', color='blue')
plt.title("Random Forest PNF R2 %98.7")
plt.xlabel("GUN")
plt.ylabel("PNF mgP-L")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()
"""
"""
x_test_CODNF = x_test_CODNF.sort_index()
y_test_CODNF = y_test_CODNF.sort_index()
"""
"""
y_test_pred_rf_CODNF = pd.DataFrame(data = y_test_pred_rf_CODNF, index = range(18), columns =['y_test_pred_rf_CODNF'])
y_test_pred_rf_CODNF = y_test_pred_rf_CODNF.sort_index()
y_test_CODNF.index = range(18)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_CODNF, label='Real Value', color='red')
ax.plot(y_test_pred_rf_CODNF, label='Predict', color='blue')
plt.title("Random Forest COD-NF R2 %89.5")
plt.xlabel("GUN")
plt.ylabel("COD-NF mg-L")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()
"""

"""
dot_data = StringIO()
export_graphviz(rf_reg_NHUF, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
"""
"""
export_graphviz(tree, out_file = 'tree8.dot', filled=True, rounded=True,
                special_characters=True)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree8.dot')
# Write graph to a png file
graph.write_png('odayeri_rf_CODNF_3.png')
"""