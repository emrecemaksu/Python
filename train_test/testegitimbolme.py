#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:27:39 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, Imputer, LabelEncoder
from sklearn.cross_validation import train_test_split

veriler = pd.read_csv('eksikveriler.csv')

yas = veriler.iloc[:, 1:4].values
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
yas = imputer.fit_transform(yas)
print(yas)
ulkeler = veriler.iloc[:, 0:1].values
le = LabelEncoder()
ulkeler[:,0] = le.fit_transform(ulkeler[:,0])
print(ulkeler)
ohe = OneHotEncoder(categorical_features="all")
ulkeler = ohe.fit_transform(ulkeler).toarray()
print(ulkeler)

sonuc = pd.DataFrame(data = ulkeler, index = range(22), columns=['fr','tr','us'])
print(sonuc)
sonuc2 = pd.DataFrame(data = yas, index = range(22), columns=['boy', 'kilo', 'yas'])
print(sonuc2)
cinsiyet = veriler.iloc[:, 4:]
print(cinsiyet)
toplam2 = pd.concat([sonuc, sonuc2], axis=1)
print(toplam2)
toplam = pd.concat([sonuc, sonuc2, cinsiyet], axis=1)
print(toplam)

x_train, x_test, y_train, y_test = train_test_split(toplam2, cinsiyet, test_size = 0.33, random_state=0)
