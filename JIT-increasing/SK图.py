from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
pandas2ri.activate()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
import sys
import re
class __Autonomy__(object):
    """ 自定义变量的write方法 """
    def __init__(self):
        """ init """
        self._buff = ""
    def write(self, out_stream):
        """ :param out_stream: :return: """
        self._buff += out_stream



plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Times New Roman'
project_path=r"./metric_result/split/within-project"
# models=["ACE_100","ACE_150","ACE_200","ACE_250","ACE_300"]
models=["within_balance_mix","within_balance_contribution","within_balance_time"]
metrics=["Acc","F1","MCC"]
dict_name={
    "within_newest":"SP",
    "within_head":"TRD",
    "within_base":"Base",
    "within_time_gauss":"TBW+gauss",
    "within_time_liner":"TBW+liner",
    "within_time_exp":"TBW+exp",
    "within_simple_weight":"CBW_without_C",
    "within_lime":"CBW_without_D",
    "within_lime_denoise":"CBW",
    "within_complex_add":"TBW_ADD_CBW",
    "within_complex_mul":"TBW_MUL_CBW",
    "within_balance_oversample":"ROS",
    "within_balance_undersample":"RUS",
    "within_balance_smote":"SMOTE",
    "within_balance_prop":"ORB",
    "within_balance_cluster":"Filtering",
    "within_balance_ensemble":"IL_ensemble",
    "within_balance_mix":"WBO",
    "within_balance_contribution":"WBO_without_T",
    "within_balance_time":"WBO_without_C",
}


dict_y={
    "Acc":[0.6,0.85],
    "F1":[0.1,0.8],
    "MCC":[0.05,0.6],
}

# projects=[]
# # projects=["ambari","ant","felix","jackrabbit","jenkins","lucene"]
# train_projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
# for train_project in train_projects:
#     test_projects = train_projects.copy()
#     test_projects.remove(train_project)
#     for test_project in test_projects:
#         projects.append("{}_{}".format(train_project,test_project))
fig, axes = plt.subplots(figsize=(19, 5),nrows=1, ncols=3,constrained_layout=True)
fig_count=0
for metric in metrics:
    current = sys.stdout
    a = __Autonomy__()
    sys.stdout = a
    metric_data=pd.DataFrame()

    # metric_data["task"]=np.array(projects)
    for model in models:
        path=project_path+"/{}.csv".format(model)

        model_data=pd.read_csv(path)
        metric_data[dict_name[model]]=model_data[metric]

    sk = importr('ScottKnottESD')
    r_sk = sk.sk_esd(metric_data,version = "np")
    print(r_sk)
    sys.stdout = current
    ranking=a._buff.split()
    print(ranking)
    dict_rank={}
    for i in range(1,len(ranking)//2+1):
        if ranking[i]==models[-1]:
            dict_rank[models[-1]]=ranking[i+len(ranking)//2]
        else:
            dict_rank[ranking[i]] = ranking[i + len(ranking) // 2]

    order=list(dict_rank.keys())
    for i in range(len(order)):
        order[i]=order[i].replace(".","+")
    metric_data=metric_data[order]


    ax = metric_data.plot.box(title="",ax=axes[fig_count % 3], showfliers=False, patch_artist=None,return_type='dict')

    base=order[0]
    count=0
    color=['r','palegreen','skyblue','darkorange']
    dict_temp={}
    for key in dict_rank:
        if "." in key:
            new_key=key.replace(".","+")
            dict_temp[new_key]=dict_rank[key]
        else:
            dict_temp[key]=dict_rank[key]
    dict_rank = dict_temp

    for i in range(0, len(order)):
        box=ax["boxes"][i]
        if dict_rank[order[i]]>dict_rank[base]:
            count+=1
        box.set(color=color[count], linewidth=2)
        base = order[i]


    base=order[0]


    for i in range(1,len(order)):
        if dict_rank[order[i]]>dict_rank[base]:

            axes[fig_count % 3].plot([i+0.5, i+0.5], dict_y[metric], c='b', linestyle='--')
        base=order[i]
    # y_major_locator=MultipleLocator(0.2)
    # ax.yaxis.set_major_locator(y_major_locator)
    axes[fig_count % 3].set_ylabel(metric,fontsize = 18)
    axes[fig_count % 3].set_xlabel("methods",fontsize = 18)
    axes[fig_count % 3].grid(linestyle="--", alpha=0.3)
    axes[fig_count % 3].tick_params(labelsize=18)
    fig_count += 1
plt.show()

