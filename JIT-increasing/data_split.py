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
from sklearn import preprocessing
import os
features_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']


def clean_data(path):
    data = pd.read_csv(path)
    data = data[data['contains_bug'].isin([True, False])]
    data['fix'] = data['fix'].map(lambda x: 1 if x > 0 else 0)
    data['contains_bug'] = data['contains_bug'].map(lambda x: 1 if x > 0 else 0)

    for feature in features_names:
        data = data[data[feature].astype(str).str.replace(".", "", 1).str.isnumeric()]

    data.dropna(axis=0, how='any')
    data = data[(data["la"] > 0) & (data["ld"] > 0)]
    data = data.reset_index(drop=True)
    return data

def get_state(data,project,date):
    # project_bar= {'ambari': 60, 'ant': 60, 'aptoide': 30, 'camel': 90, 'cassandra': 60, 'egeria':30, 'felix': 90, 'jackrabbit':90, 'jenkins': 90, 'lucene': 120}
    project_bar={'lucene': date}
    bar_count = 5
    gap_count = 5

    datatime = data[["author_date_unix_timestamp"]]
    datatime = (datatime - datatime.min()) // (24 * 60 * 60)

    data["bar"]=data["author_date_unix_timestamp"]// (24 * 60 * 60*project_bar[project])
    data["bar"] = data["bar"] - data["bar"].min()
    # data["bar"]=data["bar"].map(lambda x: x - x.min())

    temp_lst_gap = int(datatime.max() // project_bar[project]) + 1
    hist1 = np.histogram(datatime, bins=temp_lst_gap)

    df_gap = pd.DataFrame({"size": [], "start": [], "stop": []})

    for i in range(0, len(hist1[0]) - bar_count):
        df_gap.loc[len(df_gap)] = [sum(hist1[0][i:i + bar_count]), i, i + bar_count]
    df_gap = df_gap.sort_values("size", ascending=False)
    lst_state = []
    for index, row in df_gap.iterrows():
        if len(lst_state) == 0:
            lst_state.append((row["size"], row["start"], row["stop"]))
            continue
        if len(lst_state) < gap_count:
            flag = 1
            for item in lst_state:
                if (item[1] < row["start"] and row["start"] < item[2]) or (
                        item[1] < row["stop"] and row["stop"] < item[2]):
                    flag = 0
            if flag:
                lst_state.append((row["size"], row["start"], row["stop"]))
    lst_df_state=[]
    for state in lst_state:
        df_state=data[(data["bar"] >= state[1]) & (data["bar"] < state[2])]
        df_state=df_state.reset_index(drop=True)

        lst_df_state.append(df_state)


    return lst_df_state


if __name__=='__main__':

    projects =["lucene"]
    for project in projects:
        path=r"../data/traditional_data/{}.csv".format(project)
        data = clean_data(path)
        for date in [60,90,120,150,180,240]:
            if not os.path.exists(f'../data/traditional_data/lucene/{date}'):
                os.makedirs(f'../data/traditional_data/lucene/{date}')
            lst_df_state = get_state(data, project,date)
            for i in range(len(lst_df_state)):
                lst_df_state[i].to_csv(r"../data/traditional_data/lucene/{}/{}_{}.csv".format(date,project,i))
