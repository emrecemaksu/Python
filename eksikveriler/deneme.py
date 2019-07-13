#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:06:28 2018

@author: emrecemaksu
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

veriler = pd.read_csv('eksikveriler.csv')
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
yas = veriler.iloc[: , 1:4].values
yas = imputer.fit_transform(yas)
print(yas)
