#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 14:49:24 2019

@author: emrecemaksu
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import pydot


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

degerler = pd.concat([Cod, Nhdort, SS, ALF, Sicaklik, MLSSAero], axis=1)
degerler2 = pd.concat([Cod, Nhdort, ALF, Sicaklik, MLSSAero], axis=1)


x_train, x_test, y_train, y_test = train_test_split(degerler, Coduf, test_size=0.10, random_state=16)

rf_reg = RandomForestRegressor(n_estimators = 1, random_state=90)
rf_reg.fit(x_train, y_train)
y_test_pred_rf = rf_reg.predict(x_test)
print('\nRandom Forest CODUF R2 Skoru => ')
print(r2_score(y_test, y_test_pred_rf))
tree = rf_reg.estimators_[0]

#☺x_test = x_test.sort_index()
#y_test = y_test.sort_index()
y_test_pred_rf = pd.DataFrame(data = y_test_pred_rf, index = range(36), columns =['y_test_pred_rf'])
#y_test_pred_rf = y_test_pred_rf.sort_index()
y_test.index = range(36)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test, label='Real Value', color='red')
ax.plot(y_test_pred_rf, label='Predict', color='blue')
plt.title("Random Forest CODUF R2 %91.6")
plt.xlabel("GUN")
plt.ylabel("KOI-UF (mg/L)")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()


"""
dot_data = StringIO()
export_graphviz(rf_reg_NHUF, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
"""
"""
export_graphviz(tree, out_file = 'tree4.dot', filled=True, rounded=True,
                special_characters=True)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree4.dot')
# Write graph to a png file
graph.write_png('komurcuoda_rf_CODUF.png')
"""
