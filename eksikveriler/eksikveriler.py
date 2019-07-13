#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:33:56 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('eksikveriler.csv')

#eksik veriler
    #sci - kit learn
    
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
Yas = veriler.iloc[: , 1:4].values
#print(Yas)
imputer = imputer.fit(Yas[: , 1:4])
Yas[: , 1:4] = imputer.transform(Yas[: , 1:4])
print(Yas)
