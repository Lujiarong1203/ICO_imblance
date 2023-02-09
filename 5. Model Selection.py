import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost
from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix,f1_score, recall_score
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate, RandomizedSearchCV
from collections import Counter
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_train.csv")
print(data.shape, '\n', data.head(5))

y=data['fraud']
x=data.drop('fraud', axis=1)
print(x.shape, y.shape)
print(Counter(y))

random_seed=42
# K-fold
kf=KFold(n_splits=5, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in kf.split(x, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# 模型选择
# LR
score_data1=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(linear_model.LogisticRegression(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data1 = score_data1.append(pd.DataFrame({'LR': [score.mean()]}), ignore_index=True)
# print(score_data1)

# KNN
score_data2=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(KNeighborsClassifier(), x, y, cv=kf, scoring=sco)
    score_data2 = score_data2.append(pd.DataFrame({'KNN': [score.mean()]}), ignore_index=True)
# print(score_data2)

# MLP
score_data3=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(MLPClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data3 = score_data3.append(pd.DataFrame({'MLP': [score.mean()]}), ignore_index=True)
# print(score_data3)

# DT
score_data4=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(DecisionTreeClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data4 = score_data4.append(pd.DataFrame({'DT': [score.mean()]}), ignore_index=True)
# print(score_data4)

# # SVC
score_data5=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(SVC(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data5 = score_data5.append(pd.DataFrame({'SVC': [score.mean()]}), ignore_index=True)
# print(score_data5)

# # BNB
score_data6=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(BernoulliNB(), x, y, cv=kf, scoring=sco)
    score_data6 = score_data6.append(pd.DataFrame({'BNB': [score.mean()]}), ignore_index=True)
# print(score_data6)

# # PA
score_data7=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(PassiveAggressiveClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data7 = score_data7.append(pd.DataFrame({'PA': [score.mean()]}), ignore_index=True)
# print(score_data7)

# # GNB
score_data8=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(GaussianNB(), x, y, cv=kf, scoring=sco)
    score_data8 = score_data8.append(pd.DataFrame({'GNB': [score.mean()]}), ignore_index=True)
# print(score_data8)

# # Random Forest
score_data9=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(RandomForestClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data9 = score_data9.append(pd.DataFrame({'RF': [score.mean()]}), ignore_index=True)
# print(score_data9)

# # SGDClassifier
score_data10=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(SGDClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data10 = score_data10.append(pd.DataFrame({'SGD': [score.mean()]}), ignore_index=True)
# print(score_data10)

# # Adaboost
score_data11=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(AdaBoostClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data11 = score_data11.append(pd.DataFrame({'Ada': [score.mean()]}), ignore_index=True)
# print(score_data11)

# # LightGBM
score_data12=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(LGBMClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data12 =score_data12.append(pd.DataFrame({'LGBM': [score.mean()]}), ignore_index=True)
# print(score_data12)
#
# # GBDT
score_data13=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(GradientBoostingClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data13 =score_data13.append(pd.DataFrame({'GBDT': [score.mean()]}), ignore_index=True)
# print(score_data13)
#
# # xgboost
score_data14=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(XGBClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data14 = score_data14.append(pd.DataFrame({'XG': [score.mean()]}), ignore_index=True)
# print(score_data14)

# # HistGBDT
score_data15=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(HistGradientBoostingClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data15 =score_data15.append(pd.DataFrame({'HGBDT': [score.mean()]}), ignore_index=True)
# print(score_data15)

# # # Catboost
# score_data16=pd.DataFrame()
# scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
# for sco in scoring:
#     score = cross_val_score(CatBoostClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
#     score_data16 =score_data16.append(pd.DataFrame({'Cat': [score.mean()]}), ignore_index=True)
# # print(score_data16)

# 汇总分数
score_data=pd.concat([score_data1, score_data2, score_data3, score_data4,
                      score_data5, score_data6, score_data7, score_data8,
                      score_data9, score_data10, score_data11, score_data12,
                      score_data13, score_data14, score_data15],
                     axis=1)
score_Data=score_data.rename(index={0:'accuracy', 1:'precision', 2:'recall', 3:'f1', 4:'roc_auc'}).T
print(score_Data)

score_Data.to_csv(path_or_buf=r'data/data_score.csv', index=True)