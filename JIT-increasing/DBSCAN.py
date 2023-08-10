import pandas as pd
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report,matthews_corrcoef
from multiprocessing import Process
import time
from sklearn.cluster import KMeans,DBSCAN
from sklearn import preprocessing
import heapq
import os

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
            gap=(stop_gap_time-datatime.iloc[commit_count-1]["author_date_unix_timestamp"])//8
            # gap = 6 * 30 * 24 * 60 * 60

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
    maxnum = np.max(features, axis=0)
    minnum = np.min(features, axis=0)
    features = (features - minnum) / (maxnum - minnum)
    features[np.isnan(features)] = 0


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

        maxnum = np.max(features, axis=0)
        minnum = np.min(features, axis=0)
        features = (features - minnum) / (maxnum - minnum)
        features[np.isnan(features)] = 0
        old_train_labels = labels[before_gap:labels.shape[0], :]
        old_train_features = features[before_gap:labels.shape[0], :]
        new_train_labels = labels[after_gap:before_gap+1, :]
        new_train_features = features[after_gap:before_gap+1, :]

        lst_test_data.append((old_train_labels, old_train_features,new_train_labels,new_train_features))


    return lst_test_data

def data_enhancement(memory,test_features,temp_predict_y,test_labels):


    fasle_pred_features=test_features[test_labels.flatten() != temp_predict_y.flatten()]
    fasle_pred_labels=test_labels[test_labels.flatten() != temp_predict_y.flatten()]
    dict_fasle_pred={0:[],1:[]}
    dict_fasle_pred[0]=fasle_pred_features[fasle_pred_labels.flatten()==0]
    dict_fasle_pred[1] = fasle_pred_features[fasle_pred_labels.flatten() == 1]
    memory[0]=np.concatenate([dict_fasle_pred[0],memory[0]],axis=0)
    memory[1] = np.concatenate([dict_fasle_pred[1], memory[1]], axis=0)
    new_memory={0:np.empty([0,14]),1:np.empty([0,14])}
    base=2
    k_0=k_1=base
    if len(dict_fasle_pred[0]) > 0:
        k_0=base if (len(memory[1])+len(dict_fasle_pred[1])-len(memory[0]))//len(dict_fasle_pred[0])<=0 else (len(memory[1])+len(dict_fasle_pred[1])-len(memory[0]))//len(dict_fasle_pred[0])+2

        for feature in dict_fasle_pred[0]:
            distance=np.linalg.norm(memory[0] - feature, axis=1)
            index_0=heapq.nsmallest(k_0, range(len(distance)), distance.take)
            index_0=index_0[1:]
            new_features=(feature+memory[0][index_0,:])/2

            if len(new_memory[0])==0:
                new_memory[0]=new_features
            else:
                new_memory[0] = np.concatenate([new_memory[0], new_features], axis=0)
    if len(dict_fasle_pred[1])>0:
        k_1 = base if (len(memory[0]) + len(dict_fasle_pred[0]) - len(memory[1])) // len(dict_fasle_pred[1]) <= 0 else (len(
            memory[0]) + len(dict_fasle_pred[0]) - len(memory[1])) // len(dict_fasle_pred[1]) + 2

        for feature in dict_fasle_pred[1]:
            distance=np.linalg.norm(memory[1] - feature, axis=1)
            index_1=heapq.nsmallest(k_1, range(len(distance)), distance.take)
            index_1=index_1[1:]
            new_features=(feature+memory[1][index_1,:])/2
            if len(new_memory[1]) == 0:
                new_memory[1] = new_features
            else:
                new_memory[1] = np.concatenate([new_memory[1], new_features], axis=0)
    print("k:", k_0, k_1)
    print("false samples:", len(dict_fasle_pred[0]), len(dict_fasle_pred[1]))
    new_memory[0]=np.concatenate([new_memory[0],memory[0]],axis=0)
    new_memory[1] = np.concatenate([new_memory[1], memory[1]], axis=0)
    features = np.concatenate([new_memory[0], new_memory[1]], axis=0)
    labels = np.array([[0]] * len(new_memory[0]) + [[1]] * len(new_memory[1]))

    return features, labels



def denoising(memory,test_features,test_labels):
    train_memory = memory

    class_num = 10
    dict_buggy_class = {}
    dict_clean_class = {}
    for n in range(class_num):
        dict_buggy_class[n] = []
        dict_clean_class[n] = []



    buggy_kmeans = KMeans(n_clusters=class_num, n_jobs=-1)
    clean_kmeans = KMeans(n_clusters=class_num, n_jobs=-1)
    buggy_y_pred = buggy_kmeans.fit_predict(np.array(train_memory[1]))
    clean_y_pred = clean_kmeans.fit_predict(np.array(train_memory[0]))

    buggy_centers = buggy_kmeans.cluster_centers_
    clean_centers = clean_kmeans.cluster_centers_

    for n in range(class_num):
        temp = np.array(train_memory[1])[buggy_y_pred[:] == n]
        dict_buggy_class[n] = np.linalg.norm(temp - buggy_kmeans.cluster_centers_[buggy_y_pred[n], :], axis=1)


    for n in range(class_num):
        temp = np.array(train_memory[0])[clean_y_pred[:] == n]
        dict_clean_class[n] = np.linalg.norm(temp - clean_kmeans.cluster_centers_[clean_y_pred[n], :], axis=1)


    new_train_memory = {0: np.array([]), 1: np.array([])}
    for n in range(class_num):
        temp_1 = np.array(train_memory[1])[buggy_y_pred[:] == n]
        temp_0 = np.array(train_memory[0])[clean_y_pred[:] == n]
        threshold_1 = len(test_labels[test_labels.flatten() == 1]) * 2 * len(dict_buggy_class[n])//len(train_memory[1])
        threshold_0 = len(test_labels[test_labels.flatten() == 0]) * 2 * len(dict_clean_class[n])//len(train_memory[0])

        index_1 = heapq.nsmallest(threshold_1, range(len(dict_buggy_class[n])), dict_buggy_class[n].take)
        index_0 = heapq.nsmallest(threshold_0, range(len(dict_clean_class[n])), dict_clean_class[n].take)
        if n == 0:
            new_train_memory[1] = temp_1[index_1, :]
            # new_train_memory[1] = np.array(train_memory[1])
            new_train_memory[0] = temp_0[index_0, :]
        else:

            new_train_memory[1] = np.concatenate([new_train_memory[1], temp_1[index_1, :]],axis=0)
            new_train_memory[0] = np.concatenate([new_train_memory[0], temp_0[index_0, :]],axis=0)


    features=np.concatenate([new_train_memory[0],new_train_memory[1]],axis=0)
    labels=np.array([[0]]*len(new_train_memory[0])+[[1]]*len(new_train_memory[1]))

    return features,labels

def train_and_predict(project):
    test_projects = ["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
    test_projects.remove(project)
    for test_project in test_projects:
        for i in range(10):

            # file_names = os.listdir('../res/cross-project/{}'.format("KK"))
            # if '{}_{}_{}.npy'.format(project, test_project, i) in file_names:
            #     print('{}_{}_{}.npy exist'.format(project, test_project, i))
            #     continue

            train_labels, train_features = pre_process_data(r"./data/traditional_data/{}.csv".format(project))
            lst_test_data=pre_process_test_data(r"./data/traditional_data/{}.csv".format(test_project))

            predict_y=np.array([])
            count=0
            label_count=0
            memory={0:[],1:[]}
            denoising_features,denoising_labels=np.array([]),np.array([])
            mcc = 0

            for test_data in lst_test_data:

                old_train_labels, old_train_features, test_labels, test_features=test_data


                if len(test_features)==0:
                    continue
                if len(memory[0])==0:
                    new_train_labels=train_labels
                    new_train_features=train_features


                else:
                    new_train_features=denoising_features
                    new_train_labels=denoising_labels


                clf=RandomForestClassifier()
                clf.fit(new_train_features,new_train_labels)

                temp_predict_y=clf.predict_proba(test_features)
                temp_predict_y = (temp_predict_y[:, 1:] - temp_predict_y[:, 0:1]) / 2 + 0.5
                temp_predict_y = temp_predict_y.flatten()
                count+=1
                print(F"{count}/{len(lst_test_data)}")

                predict_y=np.concatenate([temp_predict_y,predict_y])
                label_count+=len(test_labels)
                print("number of predicted and actual:",len(predict_y),label_count)


                memory[0] = new_train_features[new_train_labels.flatten() == 0]
                memory[1] = new_train_features[new_train_labels.flatten() == 1]
                denoising_features, denoising_labels = denoising(memory,test_features,test_labels)
                print("train samples:",len(memory[0]), len(memory[1]))
                print("test samples:",len(test_labels[test_labels.flatten() == 0]), len(test_labels[test_labels.flatten() == 1]))

                # memory[0] = denoising_features[denoising_labels.flatten() == 0]
                # memory[1] = denoising_features[denoising_labels.flatten() == 1]
                #
                # denoising_features, denoising_labels=data_enhancement(memory,test_features,np.round(temp_predict_y),test_labels)
                denoising_features = np.concatenate([denoising_features, test_features])
                denoising_labels = np.concatenate([denoising_labels, test_labels])


                test_labels = test_labels.flatten()
                temp_predict_y = np.round(temp_predict_y)
                print('{}_{} {} {} {} {} {}\n'.format(
                    project + "_" + test_project,
                    str(i),
                    precision_score(y_true=test_labels, y_pred=temp_predict_y),
                    recall_score(y_true=test_labels, y_pred=temp_predict_y),
                    f1_score(y_true=test_labels, y_pred=temp_predict_y),
                    matthews_corrcoef(y_true=test_labels, y_pred=temp_predict_y),
                    accuracy_score(y_true=test_labels, y_pred=temp_predict_y)))

            # np.save('./res/K-means/{}_{}_{}.npy'.format(project, test_project, i), predict_y)

            predict_y = np.round(predict_y)
            test_labels, test_features = pre_process_data(r"./data/traditional_data/{}.csv".format(test_project))
            test_labels = test_labels.flatten()
            print(len(test_labels),len(predict_y))



            print('{}_{} {} {} {} {} {}\n'.format(
                project+"_"+test_project,
                str(i),
                precision_score(y_true=test_labels, y_pred=predict_y),
                recall_score(y_true=test_labels, y_pred=predict_y),
                f1_score(y_true=test_labels, y_pred=predict_y),
                matthews_corrcoef(y_true=test_labels, y_pred=predict_y),
                accuracy_score(y_true=test_labels, y_pred=predict_y)))




if __name__=='__main__':

    projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins", "lucene"]
    for project in projects:
        train_and_predict(project)