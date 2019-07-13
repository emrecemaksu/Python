#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:35:39 2018

@author: emrecemaksu
"""
#kutuphane
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#kodlar
#veri yukleme

veriler = pd.read_csv('veriler.csv')
# veriler = pd.read_csv("veriler.csv")
print(veriler)
#veri on isleme
boy = veriler[['boy']]
print(boy)
boykilo = veriler[['boy', 'kilo']]
print(boykilo)

class insan:
    boy = 150
    def kosma(self, b):
        return b + 10
ali = insan()
print(ali.boy)
print(ali.kosma(50))
l = [1, 2, 4, 8]
print(l[1])
