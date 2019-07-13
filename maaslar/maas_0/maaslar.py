#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:38:33 2018

@author: emrecemaksu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

maaslar = pd.read_csv('maaslar.csv')

maas = maaslar[['maas']].values
seviye = maaslar[['Egitim Seviyesi']].values
egitim = maaslar[['Egitim Seviyesi']].values

LR = LinearRegression()
LR = LR.fit(seviye, maas)

plt.scatter(seviye, maas, color='red')
plt.plot(seviye, LR.predict(seviye), color='blue')
plt.show()

PF  = PolynomialFeatures(degree=4)
seviye = PF.fit_transform(seviye)
LR2 = LinearRegression().fit(seviye, maas)
plt.scatter(egitim, maas, color='red')
plt.plot(seviye, LR2.predict(seviye))
plt.show()

print(LR.predict(12))
print(LR2.predict(PF.fit_transform(12)))
