import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data= pd.read_csv("data/data_1.csv")
print(data.shape)

index=data.rating_overall
print(index.value_counts())

data['rating_overall']=data['rating_overall'].apply(lambda x:
                                                    float(0) if x == 'NoEntry'
                                                    else x)

# Converts partial feature types
columns=['rating_overall', 'KYC_num']
for k in columns:
    data[k]=data[k].apply(lambda x: float(x))
print('The converted data type：', '\n', data.dtypes)

data_num=data[['rating_count', 'rating_overall', 'rating_ai']]
print(data_num.shape)
print(data_num.describe().T)

print(Counter(data['fraud']))

print(data.isnull().sum())

na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio)

# fig,axes=plt.subplots(1,1,figsize=(12,6))
# sns.set(font_scale=1.4, font='Times New Roman')
# # axes.grid(color='#909090',linestyle=':',linewidth=2)
# plt.xticks(rotation=90)
# sns.barplot(x=na_ratio.index,y=na_ratio,palette='coolwarm_r')
# plt.title('Missing Value Ratio',color=('#000000'),y=1.03)
# plt.tight_layout();
# # plt.show()
#
# sns.set(style="ticks")
# msno.matrix(data)
# plt.show()

# 填充缺失值
data['rating_overall']=data['rating_overall'].fillna(data['rating_overall'].astype(float).mean())
data['rating_ai']=data['rating_ai'].fillna(data['rating_ai'].astype(float).mean())
data['rating_count']=data['rating_count'].fillna(data['rating_count'].astype(float).mean())
data['Category_sum']=data['Category_sum'].fillna(data['Category_sum'].astype(float).mean())

print('After filling in the missing values：', '\n', data.isnull().sum())

# 划分特征和标签
x=data.drop('fraud', axis=1)
y=data.fraud
print(x.shape, y.shape)

# standardized
mm=MinMaxScaler()
X=pd.DataFrame(mm.fit_transform(x))
X.columns=x.columns
print(X.head(5))

# 2维投影
plot_2Dprojection_and_cardinality(X, y)
# plt.tick_params(labelsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
plt.legend(loc='upper right')
plt.show()

data.to_csv(path_or_buf=r'data/data_2.csv', index=None)