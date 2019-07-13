#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 19:40:15 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

veriler = pd.read_csv('eksikveriler.csv')
ulkeler = veriler.iloc[:, 0:1].values
print(ulkeler)
le = LabelEncoder()
ulkeler[:, 0] = le.fit_transform(ulkeler[:, 0])
#print(ulkeler )

ohe = OneHotEncoder(categorical_features="all")
ulkeler = ohe.fit_transform(ulkeler).toarray()
print(ulkeler)
