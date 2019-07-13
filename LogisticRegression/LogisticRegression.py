#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:47:30 2018

@author: emrecemaksu
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


veriler = pd.read_csv('veriler.csv')
X = veriler.iloc[:,1:4]
y = veriler.iloc[:,4:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

sc = StandardScaler()
X_SC_train = sc.fit_transform(X_train)
X_SC_test = sc.fit_transform(X_test)

LOGR = LogisticRegression(random_state=0)
LOGR.fit(X_SC_train, y_train)
LOGR_predict = LOGR.predict(X_SC_test)
print(LOGR_predict)
print(y_test)

cm = confusion_matrix(y_test, LOGR_predict)
print(cm)

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_SC_train, y_train)
knn_pred = knn.predict(X_SC_test)
print(knn_pred)
print(y_test)
cm2 = confusion_matrix(y_test, knn_pred)
print(cm2)
