#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 12:27:39 2018

@author: emrecemaksu
"""

#Kütüphaneler

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, Imputer, LabelEncoder, StandardScaler
from sklearn.cross_validation import train_test_split

#Veri dosyasını alıyoruz.

veriler = pd.read_csv('eksikveriler.csv')

#fit_trnsform ile NaN olanlara ort değer atıyyoruz. Variable explorerdan incele

yas = veriler.iloc[:, 1:4].values
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
yas = imputer.fit_transform(yas)
print(yas)

#Ülkeleri Label encoder ile önce 1 0 2 olarak numaralandırıyoruz. Daha sonra
#OneHotEncoder ile satır sayısı kadar 1 0 0 gibi yazıyoruz.
#Variable explorerdan ulkelere tıklayarak incele

ulkeler = veriler.iloc[:, 0:1].values
le = LabelEncoder()
ulkeler[:,0] = le.fit_transform(ulkeler[:,0])
print(ulkeler)
ohe = OneHotEncoder(categorical_features="all")
ulkeler = ohe.fit_transform(ulkeler).toarray()
print(ulkeler)

#DataFrame tipine çeviriyoruz. Concat ile birleştirip tablo yaratıyoruz. 

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

#Veriyi train_test_split fonksiyonu ile 1/3 oranında test ve eğitim verisi olarak bölüyoruz.
#StandardScaler ile Öznitelik Ölçekleme yapıyoruz.
#Variable explorerdan X_train ve X_test e tıklayarak incele

split = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(toplam2, cinsiyet, test_size = 0.33, random_state=0)
X_train = split.fit_transform(x_train)
X_test = split.fit_transform(x_test)
print(X_test)
print(X_train)
