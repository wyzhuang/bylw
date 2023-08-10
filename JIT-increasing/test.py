from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score,classification_report
import os
import numpy as np
from sklearn.utils import compute_class_weight
from multiprocessing import Process
import time

def train_and_predict(project):
    for i in range(30):
        train_Y = np.load('../train_test_set/{}_train_Y.npy'.format(project)).astype(np.float64)
        test_Y = np.load('../train_test_set/{}_test_Y.npy'.format(project)).astype(np.float64)
        weight = dict(enumerate(compute_class_weight(class_weight='balanced', classes=[0, 1],
                                                     y=train_Y.tolist())))

        train_X = np.load('../train_test_set/{}_train_X.npy'.format(project)).astype(np.float64)
        test_X = np.load('../train_test_set/{}_test_X.npy'.format(project)).astype(np.float64)
        clf = RandomForestClassifier(class_weight=weight)
        clf.fit(train_X, train_Y)
        predict_y = clf.predict_proba(test_X)
        np.save('../res/DBN_RF/{}_{}.npy'.format(project, i), predict_y)
        predict_y=(predict_y[:,1:]-predict_y[:,0:1])/2+0.5
        predict_y=predict_y.flatten()
        predict_y = np.round(predict_y)
        print('{}_{} {} {} {} {}\n'.format(
                project,
                str(i),
                precision_score(y_true=test_Y, y_pred=predict_y),
                recall_score(y_true=test_Y, y_pred=predict_y),
                f1_score(y_true=test_Y, y_pred=predict_y),
                accuracy_score(y_true=test_Y, y_pred=predict_y)))

        with open('../res/DBN_RF/res.txt', 'a+', encoding='utf-8') as f:
            f.write('{}_{} {} {} {} {}\n'.format(
                project,
                str(i),
                precision_score(y_true=test_Y, y_pred=predict_y),
                recall_score(y_true=test_Y, y_pred=predict_y),
                f1_score(y_true=test_Y, y_pred=predict_y),
                accuracy_score(y_true=test_Y, y_pred=predict_y)
            ))

def compute_prf(data):
    # p,r,f,acc
    # print(data.info())
    precision = precision_score(y_true=data["bugs"], y_pred=np.round(data["predict_y_prob"]))
    recall = recall_score(y_true=data["bugs"], y_pred=np.round(data["predict_y_prob"]))
    f1 = f1_score(y_true=data["bugs"], y_pred=np.round(data["predict_y_prob"]))
    acc=accuracy_score(y_true=data["bugs"], y_pred=np.round(data["predict_y_prob"]))
    mcc=matthews_corrcoef(y_true=data["bugs"], y_pred=np.round(data["predict_y_prob"]))
    return precision, recall, f1, acc,mcc


def compute_area(data):
    loc_sum = float(data[['loc']].apply(sum))
    bug_file_sum = float(data[['bugs']].apply(sum))
    after_array=np.cumsum(np.array(data["bugs"] / bug_file_sum ))
    before_array=np.cumsum(np.concatenate(([0],np.array(data["bugs"] / bug_file_sum )[0:len(data)-1])))
    loc_array=np.array(data["loc"] / loc_sum )
    area=np.sum((after_array+before_array) * loc_array / 2)
    return area

def compute_popt(data):
    sorted_optimal = data.sort_values(['bugs', 'loc'], inplace=False, ascending=[False, True])
    sorted_optimal['bugs'] = sorted_optimal['bugs'].map(lambda x: 1 if x > 0 else 0)
    sorted_worst = data.sort_values(['bugs', 'loc'], inplace=False, ascending=[True, False])
    sorted_worst['bugs'] = sorted_worst['bugs'].map(lambda x: 1 if x > 0 else 0)
    sorted_prediction_pre = data.sort_values('predict_y_prob', inplace=False, ascending=False)
    sorted_prediction_pre['bugs'] = sorted_prediction_pre['bugs'].map(lambda x: 1 if x > 0 else 0)
    sorted_prediction_pre['density'] = sorted_prediction_pre['predict_y_prob'] / sorted_prediction_pre['loc']
    sorted_prediction = sorted_prediction_pre.sort_values(['density'], inplace=False, ascending=False)
    compute_area(sorted_optimal)
    popt = 1 - (compute_area(sorted_optimal) - compute_area(sorted_prediction)) / (compute_area(sorted_optimal) - compute_area(sorted_worst))
    return popt

def pci_20(data):
    sorted_prediction = data.sort_values('predict_y_prob', inplace=False, ascending=False)
    # sorted_prediction_pre['bugs'] = sorted_prediction_pre['bugs'].map(lambda x: 1 if x > 0 else 0)
    # sorted_prediction_pre['density'] = sorted_prediction_pre['predict_y_prob'] / sorted_prediction_pre['loc']
    # sorted_prediction = sorted_prediction_pre.sort_values(['density'], inplace=False, ascending=False)
    loc_sum = float(data[['loc']].apply(sum))
    count=0
    locs=0
    for index, row in sorted_prediction.iterrows():
        locs+=row['loc']
        count+=1
        if locs / loc_sum > 0.2:
            return count/len(data)

def compute_pofb20(data):

    sorted_prediction = data.sort_values('predict_y_prob', inplace=False, ascending=False)

    loc_sum = np.sum(data['loc'])
    bug_sum = np.sum(data['bugs'])
    bug = 0
    locs = 0
    for index, row in sorted_prediction.iterrows():
        locs += row['loc']
        bug += row['bugs']
        if locs / loc_sum > 0.2:
            return bug / bug_sum

projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene-solr"]
metric_results=open('./metric_results.csv', 'w', encoding='utf-8')
metric_results.write('task,Precision,Recall,F1,Acc,MCC,PofB20,Popt,Pci\n')
for project in projects:

    project_path = r"../traditional_data/{}.csv".format(project)
    test_commit = np.load(r'../train_test_set/{}_{}.npy'.format(project,"test_commits"))
    test_label = np.load(r'../train_test_set/{}_{}.npy'.format(project,"test_Y")).astype(np.float_)

    data = pd.read_csv(project_path)
    data = data[data['contains_bug'].isin([True, False])]
    data['fix'] = data['fix'].map(lambda x: 1 if x > 0 else 0)
    data['contains_bug'] = data['contains_bug'].map(lambda x: 1 if x > 0 else 0)
    data=data.iloc[::-1, :].reset_index()
    test_data = np.load('../commit_guru/{}_test.npy'.format(project), allow_pickle=True)

    project_data = data[data["commit_hash"].isin(test_commit)]
    project_data = project_data.reset_index()
    label_dict= {}
    for i in range(len(test_commit)):

        label_dict[test_commit[i]]=test_label[i]

    test_commit=np.array(list(label_dict.keys()))
    test_label=np.array(list(label_dict.values())).astype(np.float_)



    project_data["commit"]=test_commit
    project_data = project_data[['ns', 'nd', 'nf', 'entropy', 'la', 'ld', 'lt', 'fix', 'ndev', 'age', 'nuc', 'exp', 'rexp', 'sexp',
                 'contains_bug']]
    project_data.rename(columns={'contains_bug': 'bugs'}, inplace=True)
    for i in range(0,30):
        file_name = '{}_{}'.format(project, str(i))

        predict_y = np.load(r'./DBN_RF/{}.npy'.format(file_name)).astype(np.float_)
        predict_y = (predict_y[:, 1:] - predict_y[:, 0:1]) / 2 + 0.5
        predict_y = predict_y.flatten()

        prob = pd.DataFrame(predict_y, columns=['predict_y_prob'])
        prob=prob[0:len(project_data)]

        bug_label=pd.DataFrame(test_label, columns=['bugs'])
        project_data["bugs"]=bug_label["bugs"]

        data = pd.concat([project_data.reset_index(), prob], axis=1)
        data["loc"] = data["la"] + data["ld"]

        precision, recall, f1, acc,mcc = compute_prf(data)
        pofb20 = compute_pofb20(data)
        popt = compute_popt(data)
        pci = pci_20(data)
        metric_results.write(
            "{},{},{},{},{},{},{},{},{}\n".format(file_name, precision, recall, f1, acc,mcc, pofb20, popt, pci))
        print(file_name, precision, recall, f1, acc,mcc, pofb20, popt, pci)
        metric_results.flush()

if __name__=='__main__':
    projects = ["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene-solr"]
    processes = [
        Process(target=train_and_predict, args=(project,)) for project in projects
    ]
    start_time = time.time()
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    end_time = time.time()
    print("done")
    print("process time: {}".format(end_time - start_time))
