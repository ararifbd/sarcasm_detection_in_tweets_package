# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:33:58 2021

@author: Arifur Rahaman
"""

import os, pandas as pd
FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\..\\features\\features.csv"
DATASET_FILE_PATH = os.curdir + "\\..\\data\\dataset.csv"
def read_data(filename):
    data = pd.read_csv(filename, header=None, encoding="utf-8", names=["Index", "Label", "Tweet"])
    data = data[data["Index"] < 5]
#    data[(data["Index"]>0) & (data["Index"]<1000)]
#    data.loc[[0,2,4]]
    #data.loc[1:3]
    return data
print(read_data(DATASET_FILE_PATH))