from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
import sys
import re

projects =["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
para_dict={}
with open('./res/within_time_para/para2.txt', 'r', encoding='utf-8') as f:
    para_dict = eval(f.readline())


lst_para=[]
lst_bar=[]
lst_sample=[]
for project in projects:
    for i in range(5):
        lst_para.append(para_dict["guass"][project+"_"+str(i)]["para"])
        lst_bar.append(para_dict["guass"][project + "_" + str(i)]["bar_size"])
        lst_sample.append(para_dict["guass"][project + "_" + str(i)]["sample_size"])
for i in range(len(lst_sample)):
    print(f'{lst_para[i]:5}',end=" ")
    print(f'{lst_bar[i]:5}',end=" ")
    print(f'{lst_sample[i]:5}',end=" ")
    print(f'{int(lst_sample[i]/lst_bar[i]):5}', end=" ")
    print()
