import numpy as np
from scipy import stats
import pandas as pd
import cliffsdelta as cliff


def p_and_delta(num1,num2):
    if num1>0.05:
        a="{:.3f}".format(num1)
    else:
        a="$<$0.05"
    b=""
    if num2<0:
        b="-"
    else:
        b="+"
    num2=abs(num2)
    if num2<0.147:
        b=b+"Negligible"
    elif 0.147<=num2<0.33:
        b = b +"Small"
    elif 0.33<=num2<0.474:
        b = b +"Medium"
    else:
        b = b +"Large"
    return "{}({})".format(a,b)

def BH(pd_data):
    global models
    dict_method ={}
    for i in range(len(models)-1):
        dict_method[models[i]]=i*3+2
    for method in models:
        if method=="ACE":
            break
        p_data=pd_data[method+"_p"]
        pd_data_sorted=pd_data.sort_values(method+"_p", inplace=False, ascending=True)
        p_data_sorted=pd_data_sorted[method+"_p"]


        for i in range(0,len(p_data_sorted)):
            pd_data_sorted.iloc[i, dict_method[method]]=pd_data_sorted.iloc[i, dict_method[method]]*len(p_data_sorted)/(i+1)
        for i in range(len(p_data_sorted) - 2, -1, -1):
            pd_data_sorted.iloc[i, dict_method[method]]=min(pd_data_sorted.iloc[i, dict_method[method]],pd_data_sorted.iloc[i+1, dict_method[method]])

        for index, row in pd_data_sorted.iterrows():
            pd_data.loc[index,method + "_p"]=pd_data_sorted.loc[index,method + "_p"]
    return  pd_data

metrics=["F1"]
models=["traditional","Deeper","DBN","DAECNN-JDP","ACE"]
dict_model_name={}
projects=[]

dict_methods={
    "traditional":"RF",
    "Deeper":"LR",
    "DAECNN-JDP":"encode",
    "DBN":"300",
    "ACE":"merge"
}

project_path=r"./metric_result/cross-project"
train_projects=["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
for train_project in train_projects:
    test_projects = ["ambari","ant","aptoide","camel","cassandra","egeria","felix","jackrabbit","jenkins","lucene"]
    test_projects.remove(train_project)
    for test_project in test_projects:
        projects.append("{}_{}".format(train_project,test_project))

for metric in metrics:
    metric_data=pd.DataFrame()
    metric_data["task"]=np.array(projects)
    for model in models:
        path=project_path+"/{}/{}.csv".format(model,model+"_"+dict_methods[model])
        model_data=pd.read_csv(path)
        lst_mean=[]
        for project in projects:
            selected_project=model_data[model_data["task"].astype(str).str.contains(project)]
            lst_mean.append(np.mean(selected_project[metric]))
        metric_data[model]=np.array(lst_mean)
        if model=="ACE":
            break
        ace_data=pd.read_csv(project_path+"/{}/{}.csv".format("ACE","ACE"+"_"+dict_methods["ACE"]))

        lst_pvalue=[]
        lst_delta=[]
        for project in projects:
            central_data=ace_data[ace_data["task"].astype(str).str.contains(project)]
            compare_data=model_data[model_data["task"].astype(str).str.contains(project)]

            lst_pvalue.append(stats.wilcoxon(central_data[metric], compare_data[metric])[1])
            lst_delta.append(cliff.cliffsDelta(central_data[metric],compare_data[metric])[0])
        metric_data[model+"_p"]=np.array(lst_pvalue)
        metric_data[model + "_delta"] = np.array(lst_delta)
    pd_data=metric_data
    pd_data_BH=BH(metric_data)
    data=metric_data.values.tolist()
    pd.set_option('display.max_columns', None)

    print(r" &\multicolumn{"+str(len(models))+"}{c|}{" + metric + r"}  &\multicolumn{"+str(len(models)-1)+r"}{c}{$p$($\delta$)}   \\\hline")
    # lst_title=["task",r"\\\hline"]
    # for i in range(1,len(models)):
    #     lst_title.insert(i,"&{}".format(models[i-1]))
    #     lst_title.insert(2*i, "&{} vs.{}".format(models[i - 1],"GH-ACE"))
    # lst_title.insert(len(models), "&{}".format("GH-ACE"))
    # print(" ".join(lst_title))
    print(
        "task &RF &QRS2015 &ICSE2016 &IET2020 &GH-ACE &RF vs.GH-ACE &QRS2015 vs.GH-ACE &ICSE2016 vs.GH-ACE &IET2020 vs.GH-ACE \\\hline")
    for i in range(0, len(projects)):
        lst_line = data[i].copy()
        lst_line[0] = lst_line[0].replace("_", "\\_")
        # lst_line[-1] = lst_line[-1].replace("\n", "")

        for j in range(1, len(models)):
            lst_line[3 * (j - 1) + 1] = round(float(lst_line[3 * (j - 1) + 1]), 3)
            lst_line[3 * (j - 1) + 2] = round(float(lst_line[3 * (j - 1) + 2]), 3)
            lst_line[3 * (j - 1) + 3] = round(float(lst_line[3 * (j - 1) + 3]), 3)
        lst_line[-1] = round(float(lst_line[-1]), 3)
        if metric == "Pci":
            lst_bold=[]
            for j in range(1, len(models)+1):
                lst_bold.append(lst_line[3*(j-1)+1])
            bold_num=min(lst_bold)
        else:
            lst_bold = []
            for j in range(1, len(models) + 1):
                lst_bold.append(lst_line[3 * (j - 1) + 1])
            bold_num = max(lst_bold)
        for j in range(1, len(models) + 1):
            if -0.00001 < lst_line[3 * (j - 1) + 1] - bold_num < 0.00001:
                lst_line[3 * (j - 1) + 1] = "\\textbf{{{:.3f}}}".format(lst_line[3 * (j - 1) + 1])
            else:
                lst_line[3 * (j - 1) + 1] = "{:.3f}".format(lst_line[3 * (j - 1) + 1])
        print("{:18}".format(lst_line[0]), end="")
        for j in range(0,len(models)):
            print("&{:10}".format(lst_line[3*j+1]), end="")
        for j in range(0,len(models)-1):
            print("&{:10}".format(p_and_delta(lst_line[3*j+2], lst_line[3*j+3])), end="")
        if i != len(projects)-1:
            print(r"\\")
        else:
            print(r"\\\hline")
    lst_avg =[]
    lst_avg.append("Average")
    for model in models:
        mean=np.mean(pd_data[model])
        lst_avg.append(round(mean,3))
    if metric == "Pci":
        bold_num = min(lst_avg[1:])
    else:
        bold_num = max(lst_avg[1:])
    for j in range(1, len(models) + 1):
        if -0.00001 < lst_avg[j] - bold_num < 0.00001:
            lst_avg[j] = "\\textbf{{{:.3f}}}".format(lst_avg[j])
        else:
            lst_avg[j] = "{:.3f}".format(lst_avg[j])
    lst_generate_wtl = []
    for i in range(1, len(models)*3-1):
        pd.to_numeric(pd_data_BH.iloc[:, i])
    for i in range(1, len(models)):
        if metric == "Pci":
            l = pd_data_BH[
                (pd_data_BH.iloc[:, 3 * (i - 1) + 2] < 0.05) & (pd_data_BH.iloc[:, 3 * (i - 1) + 3] >= 0.147)]
            w = pd_data_BH[
                (pd_data_BH.iloc[:, 3 * (i - 1) + 2] < 0.05) & (pd_data_BH.iloc[:, 3 * (i - 1) + 3] <= -0.147)]
        else:
            w = pd_data_BH[
                (pd_data_BH.iloc[:, 3 * (i - 1) + 2] < 0.05) & (pd_data_BH.iloc[:, 3 * (i - 1) + 3] >= 0.147)]
            l = pd_data_BH[
                (pd_data_BH.iloc[:, 3 * (i - 1) + 2] < 0.05) & (pd_data_BH.iloc[:, 3 * (i - 1) + 3] <= -0.147)]
        t = pd_data_BH[(pd_data_BH.iloc[:, 3 * (i - 1) + 2] >= 0.05) | (
                    (-0.147 < pd_data_BH.iloc[:, 3 * (i - 1) + 3]) & (pd_data_BH.iloc[:, 3 * (i - 1) + 3] < 0.147))]

        lst_generate_wtl.append("{}/{}/{}".format(len(w), len(t), len(l)))
    print("{:18}".format(r"Average $\&$ Win/Tie/Loss"),end="")
    for i in range(len(models)):
        print("&{:10}".format(lst_avg[i+1]),end="")
    for i in range(len(lst_generate_wtl)):
        print("&{:10}".format(lst_generate_wtl[i]),end="")
    print(r"\\")
