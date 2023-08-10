
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from inspect import signature
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

project_path=r"./metric_result/split/within-project/diff.csv"
metric="F1"
model_data = pd.read_csv(project_path)


medium_value=model_data["rate"].median()
print(medium_value)
model_data.plot.scatter(x = 'rate', y = metric, c = 'blueviolet')
plt.plot([0.1, 0.7], [0,0], c='r', linestyle='--')
#plt.plot([medium_value,medium_value], [-0.4,0.4], c='b', linestyle='--')
plt.ylabel("MCC差值", fontsize=12)
plt.xlabel("有缺陷样本占总样本比率", fontsize=12)
plt.show()