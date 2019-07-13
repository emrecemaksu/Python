#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

veriler = pd.read_csv('veriler.csv')

Yas = veriler.iloc[:,1:4].values

ulke = veriler.iloc[:,0:1].values
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()

c = veriler[['cinsiyet']].values
c[:, 0] = le.fit_transform(c[:,0])
c = ohe.fit_transform(c).toarray()
print(c)
sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = c[:, :1] , index=range(22), columns=['cinsiyet'])

s=pd.concat([sonuc,sonuc2],axis=1)
s2= pd.concat([s,sonuc3],axis=1)

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

LR = LinearRegression()
LR.fit(x_train, y_train)
y_pred = LR.predict(x_test)

boy = s[['boy']].values
sol = s.iloc[:, :3]
sag = s2.iloc[:, 4:]
toplam = pd.concat([sol, sag], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(toplam, boy, test_size=0.33, random_state=0)

LR2 = LinearRegression()
LR2.fit(X_train, Y_train)
y2_pred = LR2.predict(X_test)

X = np.append(arr = np.ones((22,1)).astype(int), values = toplam, axis = 1)
X = pd.DataFrame(data = X, index = range(22), columns=['sabit','fr','tr','us','kilo','yas','cinsiyet'])
X_l = X.iloc[:,[0,1,2,3,4,5,6]].values
r_ols = sm.OLS(endog=boy, exog=X_l).fit()
print(r_ols.summary())

X_l = X.iloc[:,[0,1,2,3,4,6]].values
r_ols = sm.OLS(endog=boy, exog=X_l).fit()
print(r_ols.summary())

X_l = X.iloc[:, [0,1,2,3,4]].values
r_ols = sm.OLS(endog=boy, exog=X_l).fit()
print(r_ols.summary())
