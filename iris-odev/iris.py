#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:26:43 2018

@author: emrecemaksu
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import svm
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import datasets
from scipy import interp

bitki = pd.read_excel('Iris.xls')
boyutlar = bitki.iloc[:,:4]
y = bitki.iloc[:, 4:5]
iris = bitki.iloc[:,4:5]

SS = StandardScaler()
SS_boyutlar = SS.fit_transform(boyutlar)
SS_boyutlar = pd.DataFrame(data=SS_boyutlar, index = range(150), columns = ['sepal length', 'sepal width', 'petal length', 'petal width'])

X_train, X_test, y_train, y_test = train_test_split(boyutlar, y, test_size=.5, random_state=0)
V_train, V_test, z_train, z_test = train_test_split(SS_boyutlar, iris, test_size=0.33, random_state=0)

#Logistic Regression
LR = LogisticRegression(random_state = 0)
LR.fit(V_train, z_train)
z_test_predict_LR = LR.predict(V_test)

print('Confusion Matrix')
cm_LR = confusion_matrix(z_test, z_test_predict_LR)
print(cm_LR)

z_test_proba_LR = LR.predict_proba(V_test)
print(z_test_proba_LR)

#DecisionTreeClassifier
DTC = DecisionTreeClassifier(criterion='entropy')
DTC.fit(V_train, z_train)
z_test_predict_DTC = DTC.predict(V_test)

print('DecisionTreeClassifier')
cm_DTC = confusion_matrix(z_test, z_test_predict_DTC)
print(cm_DTC)

z_test_proba_DTC = DTC.predict_proba(V_test)
print(z_test_proba_DTC)

#RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
RFC.fit(V_train, z_train)
z_test_predict_RFC = RFC.predict(V_test)

print('Random Forest Classifier')
cm_RFC = confusion_matrix(z_test, z_test_predict_RFC)
print(cm_RFC)

z_test_proba_RFC = RFC.predict_proba(V_test)
print(z_test_proba_RFC)

#Naive Bayes
GNB = GaussianNB()
GNB.fit(V_train, z_train)
z_test_predict_GNB = GNB.predict(V_test)

print('Naive Bayes')
cm_GNB = confusion_matrix(z_test, z_test_predict_GNB)
print(cm_GNB)

z_test_proba_GNB = GNB.predict_proba(V_test)
print(z_test_proba_GNB)

#SVC
svc = SVC(kernel='rbf')
svc.fit(V_train, z_train)
z_test_predict_svc = svc.predict(V_test)

print('SVC')
cm_SVC = confusion_matrix(z_test, z_test_predict_svc)
print(cm_SVC)

#KNN
KNN = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
KNN.fit(V_train, z_train)
z_test_predict_KNN = KNN.predict(V_test)

print('KNN')
cm_KNN = confusion_matrix(z_test, z_test_predict_KNN)
print(cm_KNN)

z_test_proba_KNN = KNN.predict_proba(V_test)
print(z_test_proba_KNN)
'''
fpr, tpr, thold = metrics.roc_curve(z_test, z_test_proba_KNN[:,0], pos_label='Iris-versicolor')
print(z_test)
print(fpr)
print(tpr)
'''
y = label_binarize(y, classes=[0, 1, 2])
n_classes = y.shape[1]

random_state = np.random.RandomState(0)
n_samples, n_features = boyutlar.shape
boyutlar = np.c_[boyutlar, random_state.randn(n_samples, 200 * n_features)]

classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
