import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.rcParams['font.family']='serif'
# plt.rcParams['font.serif']='Times New Roman'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# models=["ACE_10","ACE_20","ACE_30","ACE_40","ACE_50"]
# models=[2,3,4,5,6,8,10,12,15,20]
models=[60,90,120,150,180,240]
metrics=["F1","MCC","Acc"]
project_path=r"./metric_result/split/within-project/task"
projects=[]
# projects=["ambari","ant","felix","jackrabbit","jenkins","lucene"]
train_projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
for train_project in train_projects:
    test_projects = train_projects.copy()
    test_projects.remove(train_project)
    for test_project in test_projects:
        projects.append("{}_{}".format(train_project,test_project))

# lst_parameter=['{:.2%}'.format(1/model) for model in models]
lst_parameter=['{}月'.format(int(model/30*5)) for model in models]
# lst_parameter=['{:.1%}'.format(model/10) for model in models]
# lst_parameter=[model for model in models]

fig=plt.figure(figsize=(7, 5))
makers=["o","^","s","x",".","1","v","<",">","p","*","+","D","d","|","_"]
count=0
for metric in metrics:
    metric_data=pd.DataFrame()
    metric_data["task"]=np.array(projects)
    lst_avg = []

    plt.xlabel("不同时长的数据集")
    plt.ylabel("模型性能")
    for model in models:
        path=project_path+"/{}.csv".format(model)
        model_data=pd.read_csv(path)
        metric_data[model]=model_data[metric]
        # model_data = model_data[model_data["task"].astype(str).str.contains("_jackrabbit")]
        mean = np.mean(model_data[metric])
        lst_avg.append(mean)
    plt.plot(lst_parameter, lst_avg, label=metric, marker=makers[count])
    plt.legend(bbox_to_anchor=(1.021, 1.12), loc=1, markerscale=0.5, ncol=4, fontsize='medium')
    count+=1


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8,wspace = 0.5,hspace = 0.5)
# plt.tight_layout()
# plt.savefig("huatu.svg")
plt.show()

