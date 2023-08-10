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
import lime
import lime.lime_tabular
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import re

features_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']

def creat_gap(data):

    datatime=data[["author_date_unix_timestamp"]]
    commit_count=len(datatime)

    before_gap_time=datatime.iloc[commit_count-1]["author_date_unix_timestamp"]
    stop_gap_time=datatime.iloc[0]["author_date_unix_timestamp"]
    lst_gap=[]

    count=0
    print("months:",(stop_gap_time-before_gap_time)/(30*24*60*60))
    # gap=datatime.iloc[commit_count-1]["author_date_unix_timestamp"]-before_gap_time
    gap=(stop_gap_time-before_gap_time)//10
    while before_gap_time+gap<datatime.iloc[0]["author_date_unix_timestamp"]:
        if count!=0:
            gap=(stop_gap_time-datatime.iloc[commit_count-1]["author_date_unix_timestamp"])//10
            # gap=60*60*24*180


        after_gap_time = before_gap_time+gap
        before_gap=datatime[datatime["author_date_unix_timestamp"]>=before_gap_time].index.tolist()[-1]
        after_gap=datatime[datatime["author_date_unix_timestamp"]<after_gap_time].index.tolist()[0]
        lst_gap.append((before_gap,after_gap))
        before_gap_time= before_gap_time+ gap

        count+=1
    # lst_gap.append((after_gap-1,0))

    return lst_gap

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

def pre_process_data(path):
    data=clean_data(path)
    data = data[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug']]
    labels = np.array(data.iloc[:, [-1]])
    features = np.array(data.iloc[:, :-1])
    mean = features.mean(1).reshape(features.shape[0], 1)
    std = features.std(1).reshape(features.shape[0], 1)
    features = (features - mean) / std
    features[np.isnan(features)] = 0
    maxnum = np.max(features, axis=0)
    minnum = np.min(features, axis=0)
    features = (features - minnum) / (maxnum - minnum)
    features[np.isnan(features)] = 0
    # features = preprocessing.scale(features)
    return labels, features


def pre_process_test_data(path):
    data=clean_data(path)
    lst_gap = creat_gap(data)
    data = data[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug']]
    lst_test_data=[]
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        labels = np.array(data.iloc[:, [-1]])
        features = np.array(data.iloc[:, :-1])
        mean = features.mean(1).reshape(features.shape[0], 1)
        std = features.std(1).reshape(features.shape[0], 1)
        features = (features - mean) / std
        features[np.isnan(features)] = 0
        maxnum = np.max(features, axis=0)
        minnum = np.min(features, axis=0)
        features = (features - minnum) / (maxnum - minnum)
        features[np.isnan(features)] = 0
        # features = preprocessing.scale(features)
        old_train_labels = labels[before_gap:labels.shape[0], :]
        old_train_features = features[before_gap:labels.shape[0], :]
        new_train_labels = labels[after_gap:before_gap+1, :]
        new_train_features = features[after_gap:before_gap+1, :]

        lst_test_data.append((old_train_labels, old_train_features,new_train_labels,new_train_features))


    return lst_test_data

def liner_time_weight(path,sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence=np.array(data[["author_date_unix_timestamp"]])
    lst_time_sequence=[]
    sigma=sigma
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        train_time_sequence=1/(1+sigma*(T-train_time_sequence))
        lst_time_sequence.append(train_time_sequence)
    return lst_time_sequence

def guass_time_weight(path,sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence = np.array(data[["author_date_unix_timestamp"]])
    time_sequence=time_sequence-np.min(time_sequence, axis=0)
    lst_time_sequence = []
    # sigma=24*60*60*20
    T_max=1
    T_min=0
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)

        train_time_sequence=1/(np.sqrt(2*math.pi)*sigma)*np.power(math.e,(train_time_sequence-T)**2/(-2*sigma**2))
        lst_time_sequence.append((T_max-T_min)*train_time_sequence/train_time_sequence[0]+T_min)

    return lst_time_sequence

def exp_time_weight(path,sigma):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence = np.array(data[["author_date_unix_timestamp"]])
    time_sequence=time_sequence-np.min(time_sequence, axis=0)
    lst_time_sequence = []
    sigma=1/sigma
    T_max=1
    T_min=0
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)

        train_time_sequence=T_min+(T_max-T_min)*np.power(math.e,sigma*(train_time_sequence-T))
        lst_time_sequence.append(train_time_sequence/train_time_sequence[0])

    return lst_time_sequence

def greedy_search(explainer,clf):
    dict_exp={}
    dict_feature={}
    for feature in features_names:
        dict_exp[feature]={}
        dict_feature[feature] = 0
    struct_feature=np.zeros(14)
    while np.min(struct_feature)<1:
        list_exp=explainer.explain_instance(struct_feature, clf.predict_proba, num_features=14).as_list()
        for item in list_exp:
            list_item=item[0].split(" ")
            if len(list_item)==3:

                if list_item[1] == "<=":
                    dict_exp[list_item[0]][(0,float(list_item[2]))]=item[1]
                    dict_feature[list_item[0]]=float(list_item[2])+0.01
                if list_item[1] == ">":
                    dict_exp[list_item[0]][(float(list_item[2])), 1] = item[1]
                    dict_feature[list_item[0]] = 1
            if len(list_item)==5:
                dict_exp[list_item[2]][(float(list_item[0]), float(list_item[4]))] = item[1]
                dict_feature[list_item[2]] = float(list_item[4])+0.01
        struct_feature=np.array(list(dict_feature.values()))
    return dict_exp

def get_new_sequence(temp_predict_y,test_labels,dict_exp,test_features):
    new_hard_sequence=np.array([0.5]*len(test_labels))
    list_value=[]
    list_index=[]
    for i in range(0,len(test_labels)):
        if test_labels[i] != temp_predict_y[i]:
            value=0
            for j in range(0,14):
                for map in dict_exp[features_names[j]].keys():
                    if map[0]<=test_features[i][j] and test_features[i][j]<= map[1]:
                        value+=dict_exp[features_names[j]][map]
            list_value.append(value)
            list_index.append(i)
    np_value=np.array(list_value)
    np_value=np.absolute(np_value)
    maxnum = np.max(np_value, axis=0)
    minnum = np.min(np_value, axis=0)
    np_value = (np_value - minnum) / (maxnum - minnum)*8+1
    np_value[np.isnan(np_value)] = 0
    list_value=np_value.tolist()
    for i in range(len(list_index)):
        new_hard_sequence[list_index[i]]=list_value[i]
    return new_hard_sequence


def train_and_predict(project,para_dict):
    path=r"./data/traditional_data/{}.csv".format(project)
    sigma=para_dict["guass"][0]
    for i in range(1):
        all_data=pre_process_test_data(path)

        #all_time_sequence=guass_time_weight(path,24*60*60*sigma/100*project_bar[project.split("_")[0]])
        all_time_sequence = guass_time_weight(path, 24 * 60 * 60*10)
        predict_y = np.array([])
        hard_sequence = np.array([])
        label_count = 0
        all_test_labels= np.array([])
        lst_mcc=[]
        for count in range(0,len(all_data)):
            if count==0:
                continue
            train_labels, train_features, test_labels, test_features = all_data[count]

            if len(test_features) == 0:
                continue

            new_train_labels = train_labels
            new_train_features = train_features
            if len(hard_sequence) == 0:
                hard_sequence=np.array([0.5]*len(new_train_labels))


            # weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
            #                                               y=train_labels.transpose().tolist()[0])))
            # print(weight)
            clf=linear_model.LogisticRegression(solver='liblinear',max_iter=1000)
            # clf=RandomForestClassifier()
            explainer = lime.lime_tabular.LimeTabularExplainer(new_train_features, feature_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug'],class_names=[0, 1], discretize_continuous=True)


            clf.fit(new_train_features,new_train_labels.ravel(),sample_weight=hard_sequence)

            dict_exp=greedy_search(explainer,clf)


            temp_predict_y=clf.predict_proba(test_features)

            temp_predict_y = (temp_predict_y[:, 1:] - temp_predict_y[:, 0:1]) / 2 + 0.5
            temp_predict_y = temp_predict_y.flatten()


            print(F"{count}/{len(all_data)}")

            predict_y = np.concatenate([temp_predict_y, predict_y])
            label_count+=len(test_labels)
            print(len(predict_y),label_count)
            print("test samples:", len(test_labels[test_labels.flatten() == 0]),
                  len(test_labels[test_labels.flatten() == 1]))
            test_labels = test_labels.flatten()
            all_test_labels=np.concatenate([test_labels, all_test_labels])
            temp_predict_y = np.round(temp_predict_y)
            new_hard_sequence = get_new_sequence(temp_predict_y, test_labels, dict_exp, test_features)
            hard_sequence=np.concatenate([new_hard_sequence,hard_sequence])

            lst_mcc.append(matthews_corrcoef(y_true=test_labels, y_pred=temp_predict_y))
            print('{}_{} {} {} {} {} {}\n'.format(
                project,
                str(i),
                precision_score(y_true=test_labels, y_pred=temp_predict_y),
                recall_score(y_true=test_labels, y_pred=temp_predict_y),
                f1_score(y_true=test_labels, y_pred=temp_predict_y),
                matthews_corrcoef(y_true=test_labels, y_pred=temp_predict_y),
                accuracy_score(y_true=test_labels, y_pred=temp_predict_y)))

        save_y=np.concatenate([predict_y.reshape(1,len(predict_y)),all_test_labels.reshape(1,len(predict_y))])
        np.save('./res/within_time/{}_{}.npy'.format(project, i), save_y)
        print(lst_mcc)

        predict_y = np.round(predict_y)
        test_labels=all_test_labels
        print(len(all_test_labels),len(predict_y))

        print('{}_{} {} {} {} {} {}\n'.format(
            project,
            str(i),
            precision_score(y_true=test_labels, y_pred=predict_y),
            recall_score(y_true=test_labels, y_pred=predict_y),
            f1_score(y_true=test_labels, y_pred=predict_y),
            matthews_corrcoef(y_true=test_labels, y_pred=predict_y),
            accuracy_score(y_true=test_labels, y_pred=predict_y)))

        with open('./res/within_time/res.txt', 'a+', encoding='utf-8') as f:
            f.write('{}_{} {} {} {} {}\n'.format(project, str(i),
                precision_score(y_true=test_labels, y_pred=predict_y), recall_score(y_true=test_labels, y_pred=predict_y),
                f1_score(y_true=test_labels, y_pred=predict_y), accuracy_score(y_true=test_labels, y_pred=predict_y)))



if __name__=='__main__':
    project_bar = {'ambari': 60, 'ant': 60, 'aptoide': 30, 'camel': 90, 'cassandra': 60, 'egeria': 30, 'felix': 90,
                   'jackrabbit': 90, 'jenkins': 90, 'lucene': 120}

    with open('./res/within_time_para/para.txt', 'r', encoding='utf-8') as f:
        para_dict=f.readline()
        if para_dict =="":
            para_dict={"guass":{},"liner":{},"exp":{}}
        else:
            para_dict=eval(para_dict)
    projects =["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
    # projects = ["ambari"]
    for project in projects:
        for i in range(5):
            train_and_predict(project+"_"+str(i),para_dict)
