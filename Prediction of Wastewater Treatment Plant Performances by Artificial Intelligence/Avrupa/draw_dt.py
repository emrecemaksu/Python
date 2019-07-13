# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:51:23 2019

@author: chuck
"""

import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

veriler = pd.read_csv("veriler.csv")

Fosfat = veriler[["Fosfat"]]
Fosfat_Nan = veriler[["Fosfat"]]
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
degerler2 = pd.concat([Cod, Nhdort, Ph, Sicaklik, Bioxaerobic, Bioanoxic, Phbaer, Phbanox], axis=1)
degerler3 = pd.concat([Cod, Nhdort, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Nhdortuf, Podortuf], axis=1)
degerler4 = pd.concat([Cod, Bioxaerobic, Bioanoxic, Phbaer, Phbanox, Coduf, Nhdortuf, Podortuf], axis=1)
"""
x_train_POUF, x_test_POUF, y_train_POUF, y_test_POUF = train_test_split(degerler, Podortuf, test_size=0.25, random_state=36)
r_dt_POUF = DecisionTreeRegressor(random_state=42)
r_dt_POUF.fit(x_train_POUF,y_train_POUF)
y_test_pred_dt_POUF = r_dt_POUF.predict(x_test_POUF)
print('\nDecision Tree PO4UF R2 Skoru => ')
print(r2_score(y_test_POUF, y_test_pred_dt_POUF))


y_test_pred_dt_POUF = pd.DataFrame(data = y_test_pred_dt_POUF, index = range(44), columns =['y_test_pred_dt_POUF'])
y_test_pred_dt_POUF = y_test_pred_dt_POUF.sort_index()
y_test_POUF.index = range(44)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_POUF, label='Real Value', color='red')
ax.plot(y_test_pred_dt_POUF, label='Predict', color='blue')
#plt.plot(y_test_POUF, color='red')
#plt.plot(y_test_pred_dt_POUF, color = 'blue')
plt.title("Decision Tree PO4UF R2 %89.1")
plt.xlabel("GUN")
plt.ylabel("PO4UF mgP-L")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()
"""

x_train_Fosfatsiz, x_test_Fosfatsiz, y_train_Fosfatsiz, y_test_Fosfatsiz = train_test_split(degerler2, Coduf, test_size=0.10, random_state=122)
r_dt = DecisionTreeRegressor(random_state=189)
r_dt.fit(x_train_Fosfatsiz,y_train_Fosfatsiz)
y_test_pred_dt = r_dt.predict(x_test_Fosfatsiz)
print('\nDecision Tree R2 Skoru Fosfatsız CODUF => ')
print(r2_score(y_test_Fosfatsiz, y_test_pred_dt))

"""
x_test_POUF = x_test_POUF.sort_index()
y_test_POUF = y_test_POUF.sort_index()
"""
y_test_pred_dt = pd.DataFrame(data = y_test_pred_dt, index = range(18), columns =['y_test_pred_dt'])
y_test_pred_dt = y_test_pred_dt.sort_index()
y_test_Fosfatsiz.index = range(18)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_test_Fosfatsiz, label='Real Value', color='red')
ax.plot(y_test_pred_dt, label='Predict', color='blue')
#plt.plot(y_test_POUF, color='red')
#plt.plot(y_test_pred_dt_POUF, color = 'blue')
plt.title("Decision Tree CODUF R2 %83.4")
plt.xlabel("GUN")
plt.ylabel("CODUF mg-L")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
plt.show()


"""
dot_data = StringIO()

export_graphviz(r_dt_POUF, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png('odayeri_dt_PO4UF.png')
"""