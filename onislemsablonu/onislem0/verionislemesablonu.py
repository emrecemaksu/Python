#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split

veriler = pd.read_csv('veriler.csv')

Yas = veriler.iloc[:,1:4].values

ulke = veriler.iloc[:,0:1].values
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])

ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22), columns=['cinsiyet'])

s=pd.concat([sonuc,sonuc2],axis=1)
s2= pd.concat([s,sonuc3],axis=1)

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)