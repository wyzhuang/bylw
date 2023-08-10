import pandas as pd
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report
from multiprocessing import Process
import time
features_names=['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp']
memory_space=3000
def creat_gap(path):
    data = pd.read_csv(path)
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

        after_gap_time = before_gap_time+gap
        before_gap=datatime[datatime["author_date_unix_timestamp"]>=before_gap_time].index.tolist()[-1]
        after_gap=datatime[datatime["author_date_unix_timestamp"]<after_gap_time].index.tolist()[0]
        lst_gap.append((before_gap,after_gap))
        before_gap_time= before_gap_time+ gap

        count+=1
    lst_gap.append((after_gap-1,0))
    print(len(data))


    return lst_gap

def clean_data(path):
    data = pd.read_csv(path)
    data = data[data['contains_bug'].isin([True, False])]
    data['fix'] = data['fix'].map(lambda x: 1 if x > 0 else 0)
    data['contains_bug'] = data['contains_bug'].map(lambda x: 1 if x > 0 else 0)

    for feature in features_names:
        data = data[data[feature].astype(str).str.replace(".", "", 1).str.isnumeric()]
    data = data[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug']]

    data.dropna(axis=0, how='any')

    return data

def pre_process_data(path):
    data=clean_data(path)

    labels = np.array(data.iloc[:, [-1]])
    features = np.array(data.iloc[:, :-1])
    mean = features.mean(1).reshape(features.shape[0], 1)
    std = features.std(1).reshape(features.shape[0], 1)
    features = (features - mean) / std
    features[np.isnan(features)] = 0

    return labels, features


def pre_process_test_data(path):
    data=clean_data(path)

    lst_gap = creat_gap(path)
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

        old_train_labels = labels[before_gap:labels.shape[0], :]
        old_train_features = features[before_gap:labels.shape[0], :]
        new_train_labels = labels[after_gap:before_gap+1, :]
        new_train_features = features[after_gap:before_gap+1, :]

        lst_test_data.append((old_train_labels, old_train_features,new_train_labels,new_train_features))


    return lst_test_data

def train_and_predict(project):
    test_projects = ["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
    test_projects.remove(project)
    for test_project in test_projects:
        for i in range(10):
            train_labels, train_features = pre_process_data(r"./data/traditional_data/{}.csv".format(project))
            lst_test_data=pre_process_test_data(r"./data/traditional_data/{}.csv".format(test_project))

            predict_y=np.array([])
            count=0
            label_count=0
            for test_data in lst_test_data:

                old_train_labels, old_train_features, test_labels, test_features=test_data
                if len(test_features)==0:
                    continue
                new_train_labels=train_labels
                new_train_features=train_features
                new_train_labels=np.concatenate([new_train_labels,old_train_labels])
                new_train_features=np.concatenate([new_train_features, old_train_features])
                if len(old_train_labels)>memory_space:
                    new_train_labels=new_train_labels[len(new_train_labels)-3000:len(new_train_labels)]
                    new_train_features=new_train_features[len(new_train_features)-3000:len(new_train_features)]

                weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                             y=new_train_labels.transpose().tolist()[0])))

                clf=RandomForestClassifier()
                clf.fit(new_train_features,new_train_labels)


                temp_predict_y=clf.predict_proba(test_features)
                temp_predict_y = (temp_predict_y[:, 1:] - temp_predict_y[:, 0:1]) / 2 + 0.5
                temp_predict_y = temp_predict_y.flatten()
                count+=1
                print(F"{count}/{len(lst_test_data)}")

                predict_y=np.concatenate([temp_predict_y, predict_y])
                label_count+=len(test_labels)
                print(len(predict_y),label_count)


            # np.save('../res/newest/{}_{}_{}.npy'.format(project, test_project, i), predict_y)

            predict_y = np.round(predict_y)
            test_labels, test_features = pre_process_data(r"./data/traditional_data/{}.csv".format(test_project))
            test_labels = test_labels.flatten()
            print(len(test_labels),len(predict_y))



            print('{}_{} {} {} {} {}\n'.format(project + "_" + test_project, str(i),
                precision_score(y_true=test_labels, y_pred=predict_y),
                recall_score(y_true=test_labels, y_pred=predict_y), f1_score(y_true=test_labels, y_pred=predict_y),
                accuracy_score(y_true=test_labels, y_pred=predict_y)))


if __name__=='__main__':

    projects = ["ambari", "ant", "aptoide", "camel", "cassandra", "egeria", "felix", "jackrabbit", "jenkins", "lucene"]
    for project in projects:
        train_and_predict(project)
