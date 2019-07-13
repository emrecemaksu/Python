#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 13:47:53 2018

@author: emrecemaksu
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

veriler = pd.read_csv('veriler.csv')
ulke = veriler[['ulke']].values
cinsiyet = veriler[['cinsiyet']]
LE = LabelEncoder()
ulke[:,0] = LE.fit_transform(ulke[:,0])
OHE = OneHotEncoder(categorical_features="all")
ulke = OHE.fit_transform(ulke).toarray()
bky = veriler[['boy','kilo','yas']]
SS = StandardScaler()
bky = SS.fit_transform(bky)
ulke = pd.DataFrame(data =ulke, index = range(22), columns = ['us', 'fr', 'tr'])
bky = pd.DataFrame(data =bky, index = range(22), columns = ['boy', 'kilo', 'yas'])
ubky = pd.concat([ulke, bky], axis=1)
ubky = SS.fit_transform(ubky)
X_train, X_test, y_train, y_test = train_test_split(ubky, cinsiyet, test_size=0.33, random_state=1)

GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_test_predict = GNB.predict(X_test)
print(y_test_predict)

cm = confusion_matrix(y_test, y_test_predict)
print(cm)

DTC = DecisionTreeClassifier(criterion = 'entropy')
DTC.fit(X_train, y_train)
y_test_predict2 = DTC.predict(X_test)

cm2 = confusion_matrix(y_test, y_test_predict2)
print('DTC')
print(cm2)

RFC = RandomForestClassifier(n_estimators=10, criterion='entropy')
RFC.fit(X_train, y_train)
y_test_predict3 = RFC.predict(X_test)

cm3 = confusion_matrix(y_test, y_test_predict2)
print('RFC')
print(cm3)


y_proba = GNB.predict_proba(X_test)
print(y_proba)
y_proba2 = DTC.predict_proba(X_test)
print('DTC')
print(y_proba2)
y_proba3 = RFC.predict_proba(X_test)
print('RFC')
print(y_test)
print(y_proba3[:,0])
print(y_proba3)

fpr, tpr, thold = metrics.roc_curve(y_test, y_proba3[:,0], pos_label='e')
print(fpr)
print(tpr)
