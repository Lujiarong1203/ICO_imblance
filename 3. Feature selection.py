import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import RFECV

# import scikitplot as skplt
# plt.rc('font',family='Times New Roman')
# plt.rcParams['font.sans-serif']=['SimHei']

config={"font.family": 'serif',
        "font.size": 15,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun']}

plt.rcParams.update(config)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_2.csv")
print(data.shape)

print(Counter(data['fraud']))

random_seed=1234

# 相关性分析
# 相关性热力图
plt.rcParams['axes.unicode_minus']=False

data_corr=data[['github','facebook','twitter','telegram',
            'bitcointalk', 'linkedin','medium','reddit',
            'About','rating_count','rating_overall',
            'KYC', 'Registration', 'Premium', 'Category_sum',
            'Accepting', 'Platform', 'rating_ai', 'KYC_num', 'Whitelist',
            'fraud']]
print(data_corr.shape)

corr=data_corr.corr()
print(corr)
mask=np.triu(np.ones_like(corr, dtype=np.bool_))
fig=plt.figure(figsize=(10, 12))
ax=sns.heatmap(corr, mask=mask, fmt=".2f", cmap='gist_heat', cbar_kws={"shrink": .8},
            annot=True, linewidths=1, annot_kws={"fontsize":8})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
plt.xticks(fontproperties='Times New Roman', fontsize=8, rotation=40)
plt.yticks(fontproperties='Times New Roman', fontsize=8, rotation=40)
plt.show()

# # RFECV特征选择
# 划分数据集
x=data.drop(labels=['fraud', 'KYC'], axis=1)
y=data.fraud
print(x.shape, y.shape)

# 构造K折交叉验证
stra_kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in stra_kf.split(x, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

RF=RandomForestClassifier(random_state=random_seed)

rfecv=RFECV(estimator=RF, step=1, cv=stra_kf, scoring="f1")

rfecv.fit(x, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print("Ranking of features : %s" % rfecv.ranking_)
print("Scores of features : %s" % np.mean(rfecv.grid_scores_, axis=1))

# Plot number of features VS. cross-validation scores
plt.figure()
plt.plot(range(1, len(rfecv.grid_scores_) + 1), np.mean(rfecv.grid_scores_, axis=1), color='b')
plt.xlabel("选择特征数量", fontsize=15)
plt.ylabel("交叉验证得分", fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

print(rfecv.get_feature_names_out())

data_sel=data[rfecv.get_feature_names_out()]
print(data_sel.head(5), data_sel.shape)

data_3=pd.concat([data_sel, data['KYC'], data['fraud']], axis=1)
print(data_3.head(5), data_3.shape)

data_3.to_csv(path_or_buf=r'data/data_3.csv', index=None)