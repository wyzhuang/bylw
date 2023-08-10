import pandas as pd
import numpy as np
import math
from sklearn.utils import compute_class_weight
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report,matthews_corrcoef
from multiprocessing import Process
import time
import heapq
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pylab import mpl
import pylab

project="camel"
mpl.rcParams['font.sans-serif'] = ['SimHei']

path=r"./data/traditional_data/{}.csv".format(project)
data = pd.read_csv(path)
data=data.iloc[::-1]
datatime=data[["author_date_unix_timestamp"]]

date_count=90
bar_count=5

datatime=(datatime-np.array(datatime)[0])//(24*60*60)


temp_lst_gap=np.array(datatime)[-1][0]//date_count+1

hist1=np.histogram(datatime, bins=temp_lst_gap)

df_gap=pd.DataFrame({"size":[],"start":[],"stop":[]})
for i in range(0,len(hist1[0])-bar_count):
    df_gap.loc[len(df_gap)]=[sum(hist1[0][i:i+bar_count]),i,i+bar_count]

df_gap=df_gap.sort_values("size", ascending=False)
lst_gap=[]
for index, row in df_gap.iterrows():
    if len(lst_gap) == 0:
        lst_gap.append((row["size"], row["start"], row["stop"]))
        continue
    if len(lst_gap)<3:
        flag=1
        for item in lst_gap:
            if (item[1]<row["start"] and row["start"]<item[2]) or  (item[1]<row["stop"] and row["stop"]<item[2]):
                flag=0
        if flag:
            lst_gap.append((row["size"], row["start"], row["stop"]))



print(lst_gap)
plt.hist(np.array(datatime), bins=temp_lst_gap, rwidth=0.9, density=False)

lst_count=[0]*(np.max(np.array(datatime)//date_count)+1)

# print(np.array(datatime).reshape(1,len(datatime)).tolist())
for i in np.array(datatime)//date_count:
    lst_count[i[0]]+=1

x=range(30,np.max(np.array(datatime))+31,date_count)


plt.plot(hist1[1][1:]-np.max(np.array(datatime))/temp_lst_gap/2, hist1[0], marker='*',label="折线")
plt.title(project+'项目直方图，'+"每个直方范围："+str(date_count)+"天")
plt.xlabel('天数')
plt.ylabel('样本数量')


plt.show()

