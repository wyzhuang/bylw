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

features_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']

def creat_gap(data):

    datatime=data[["author_date_unix_timestamp"]]
    commit_count=len(datatime)

    before_gap_time=datatime.iloc[commit_count-1]["author_date_unix_timestamp"]
    stop_gap_time=datatime.iloc[0]["author_date_unix_timestamp"]
    lst_gap=[]

    count=0
    print("months:",(stop_gap_time-before_gap_time)/(30*24*60*60))
    gap=datatime.iloc[commit_count-500]["author_date_unix_timestamp"]-before_gap_time
    while before_gap_time+gap<datatime.iloc[200]["author_date_unix_timestamp"]:
        if count!=0:
            gap=(stop_gap_time-datatime.iloc[commit_count-1]["author_date_unix_timestamp"])//20
            # gap=60*60*24*180


        after_gap_time = before_gap_time+gap
        before_gap=datatime[datatime["author_date_unix_timestamp"]>=before_gap_time].index.tolist()[-1]
        after_gap=datatime[datatime["author_date_unix_timestamp"]<after_gap_time].index.tolist()[0]
        lst_gap.append((before_gap,after_gap))
        before_gap_time= before_gap_time+ gap

        count+=1
    lst_gap.append((after_gap-1,0))
    print(len(data))
    print(lst_gap)


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
    # mean = features.mean(1).reshape(features.shape[0], 1)
    # std = features.std(1).reshape(features.shape[0], 1)
    # features = (features - mean) / std
    # features[np.isnan(features)] = 0
    # maxnum = np.max(features, axis=0)
    # minnum = np.min(features, axis=0)
    # features = (features - minnum) / (maxnum - minnum)
    # features[np.isnan(features)] = 0
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
        # mean = features.mean(1).reshape(features.shape[0], 1)
        # std = features.std(1).reshape(features.shape[0], 1)
        # features = (features - mean) / std
        # features[np.isnan(features)] = 0
        # maxnum = np.max(features, axis=0)
        # minnum = np.min(features, axis=0)
        # features = (features - minnum) / (maxnum - minnum)
        # features[np.isnan(features)] = 0
        # features = preprocessing.scale(features)
        old_train_labels = labels[before_gap:labels.shape[0], :]
        old_train_features = features[before_gap:labels.shape[0], :]
        new_train_labels = labels[after_gap:before_gap+1, :]
        new_train_features = features[after_gap:before_gap+1, :]

        lst_test_data.append((old_train_labels, old_train_features,new_train_labels,new_train_features))


    return lst_test_data


def liner_time_weight(path):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence=np.array(data[["author_date_unix_timestamp"]])
    lst_time_sequence=[]
    alpha=1/(24*60*60*180)
    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        train_time_sequence=1/(1+alpha*(T-train_time_sequence))
        lst_time_sequence.append(train_time_sequence)
    return lst_time_sequence

def guass_time_weight(path):
    data = clean_data(path)
    lst_gap = creat_gap(data)
    time_sequence = np.array(data[["author_date_unix_timestamp"]])
    time_sequence=time_sequence-np.min(time_sequence, axis=0)
    lst_time_sequence = []
    sigma=24*60*60*1800

    for gap in lst_gap:
        before_gap=gap[0]
        after_gap=gap[1]
        train_time_sequence=time_sequence[before_gap:time_sequence.shape[0], :]
        T=np.max(train_time_sequence, axis=0)
        print(T)
        train_time_sequence=1/(np.sqrt(2*math.pi)*sigma)*np.power(math.e,(train_time_sequence-T)**2/(-2*sigma**2))
        lst_time_sequence.append(train_time_sequence/train_time_sequence[0])
    return lst_time_sequence



def train_and_predict(project):
    path=r"./data/traditional_data/{}.csv".format(project)

    for i in range(10):
        data = clean_data(path)
        labels, features = pre_process_data(path)
        lst_gap=creat_gap(data)
        predict_y = np.array([])
        all_test_labels = np.array([])
        print(lst_gap)
        for j in range(len(lst_gap)):
            temp_lst_gap=lst_gap.copy()
            test_gap=temp_lst_gap[j]
            temp_lst_gap.remove(temp_lst_gap[j])
            test_features=features[test_gap[1]:test_gap[0], :]
            test_labels=labels[test_gap[1]:test_gap[0], :]
            train_features=np.delete(features, np.s_[test_gap[1]: test_gap[0]], axis=0)
            train_labels = np.delete(labels, np.s_[test_gap[1]: test_gap[0]], axis=0)
            print("train_length",len(train_features))
            print("test_length",test_gap[0]-test_gap[1])
            print("",)
            clf = linear_model.LogisticRegression(solver='liblinear')
            # clf=RandomForestClassifier()
            clf.fit(train_features, train_labels)

            temp_predict_y=clf.predict_proba(test_features)

            temp_predict_y = (temp_predict_y[:, 1:] - temp_predict_y[:, 0:1]) / 2 + 0.5
            temp_predict_y = temp_predict_y.flatten()


            predict_y = np.concatenate([temp_predict_y, predict_y])

            test_labels = test_labels.flatten()
            all_test_labels=np.concatenate([test_labels, all_test_labels])
            temp_predict_y = np.round(temp_predict_y)
            print('{}_{} {} {} {} {} {}\n'.format(
                project,
                str(i)+"_"+str(j),
                precision_score(y_true=test_labels, y_pred=temp_predict_y),
                recall_score(y_true=test_labels, y_pred=temp_predict_y),
                f1_score(y_true=test_labels, y_pred=temp_predict_y),
                matthews_corrcoef(y_true=test_labels, y_pred=temp_predict_y),
                accuracy_score(y_true=test_labels, y_pred=temp_predict_y)))


        save_y=np.concatenate([predict_y.reshape(1,len(predict_y)),all_test_labels.reshape(1,len(predict_y))])
        np.save('./res/within_time/{}_{}.npy'.format(project, i), save_y)


        predict_y = np.round(predict_y)
        test_labels=all_test_labels
        print(len(all_test_labels),len(predict_y))

        print('{}_{} {} {} {} {} {}\n'.format(
            project,
            str(j),
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

    projects =["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
    for project in projects:
        train_and_predict(project)
