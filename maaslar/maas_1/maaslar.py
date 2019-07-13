#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:38:33 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

maaslar = pd.read_csv('maaslar.csv')

maas = maaslar[['maas']].values
seviye = maaslar[['Egitim Seviyesi']].values
egitim = maaslar[['Egitim Seviyesi']].values

LR = LinearRegression().fit(seviye, maas)

plt.scatter(seviye, maas, color='red')
plt.plot(seviye, LR.predict(seviye))
plt.show()

PF = PolynomialFeatures(degree=4)
seviye = PF.fit_transform(seviye)
LR2 = LinearRegression().fit(seviye, maas)

plt.scatter(egitim, maas)
plt.plot(seviye, LR2.predict(seviye))
plt.show()

SC = StandardScaler()
egitim = SC.fit_transform(egitim)
maas = SC.fit_transform(maas)

svr = SVR(kernel='rbf')
svr.fit(egitim, maas)
tahmin = svr.predict(egitim)
plt.scatter(egitim, maas)
plt.plot(egitim, tahmin)