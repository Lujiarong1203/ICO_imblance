import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import scikitplot as skplt

# plt.rc('font',family='Times New Roman')
# plt.rcParams['font.sans-serif']=['SimHei']

config={"font.family": 'serif',
        "font.size": 15,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun']}

plt.rcParams.update(config)

# Load the data
data= pd.read_csv("data/data_train.csv")
print(data.shape, '\n', data.head(5))

y=data['fraud']
x=data.drop('fraud', axis=1)
print(x.shape, y.shape)
print(Counter(y))

random_seed=42

# Load the data
data_test=pd.read_csv('data/data_test.csv')
print(data_test.shape)

x_test=data_test.drop(['fraud'], axis=1)
y_test=data_test['fraud']
print(x.shape, y.shape, x_test.shape, y_test.shape)

# 准备比较的模型
# LR
LR=LogisticRegression(random_state=random_seed)
LR.fit(x, y)
y_pred_LR=LR.predict(x_test)
y_proba_LR=LR.predict_proba(x_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)

# SVM
SVM=SVC(probability=True, random_state=random_seed)
SVM.fit(x, y)
y_pred_SVM=SVM.predict(x_test)
y_proba_SVM=SVM.predict_proba(x_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(x, y)
y_pred_RF=RF.predict(x_test)
y_proba_RF=RF.predict_proba(x_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(x, y)
y_pred_Ada=Ada.predict(x_test)
y_proba_Ada=Ada.predict_proba(x_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(x, y)
y_pred_GBDT=GBDT.predict(x_test)
y_proba_GBDT=GBDT.predict_proba(x_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)





# KNN
KNN=KNeighborsClassifier()
KNN.fit(x, y)
y_pred_KNN=KNN.predict(x_test)
y_proba_KNN=KNN.predict_proba(x_test)
cm_KNN=confusion_matrix(y_test, y_pred_KNN)

# XG
XG=XGBClassifier(random_state=random_seed)
XG.fit(x, y)
y_pred_XG=XG.predict(x_test)
y_proba_XG=XG.predict_proba(x_test)
cm_XG=confusion_matrix(y_test, y_pred_XG)

# LGBM
LGBM=LGBMClassifier(learning_rate=0.4,
                    n_estimators=120,
                    max_depth=17,
                    num_leaves=24,
                    min_child_samples=12,
                    min_child_weight=0.001,
                    colsample_bytree=1,
                    random_state=random_seed
                    )

# Verify that the optimal parameters improve the effect
LGBM.fit(x, y)
y_pred_LGBM=LGBM.predict(x_test)
y_proba_LGBM=LGBM.predict_proba(x_test)
acc_LGBM=accuracy_score(y_test, y_pred_LGBM)
pre_LGBM=precision_score(y_test, y_pred_LGBM)
rec_LGBM=recall_score(y_test, y_pred_LGBM)
f1_LGBM=f1_score(y_test, y_pred_LGBM)
auc_LGBM=roc_auc_score(y_test, y_pred_LGBM)
cm_LGBM=confusion_matrix(y_test, y_pred_LGBM)
print('On the test set:', acc_LGBM, pre_LGBM, rec_LGBM, f1_LGBM, auc_LGBM, '\n', cm_LGBM)

# 比较混淆矩阵
# KNN
skplt.metrics.plot_confusion_matrix(y_test, y_pred_KNN, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(a) KNN', y=0.2, fontsize=15)
plt.xlabel('预测值', fontsize=15)
plt.ylabel('真实值', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# Ada
skplt.metrics.plot_confusion_matrix(y_test, y_pred_Ada, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(b) Ada', y=0.2, fontsize=15)
plt.xlabel('预测值', fontsize=15)
plt.ylabel('真实值', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()

# LGBM
skplt.metrics.plot_confusion_matrix(y_test, y_pred_LGBM, title=None, cmap='tab20_r', text_fontsize=15)
# plt.title('(c) LightGBM', y=0.2, fontsize=15)
plt.xlabel('预测值', fontsize=15)
plt.ylabel('真实值', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.show()


# ROC curve# GBDT RF
# RF
skplt.metrics.plot_roc(y_test, y_proba_RF, cmap='Set1', text_fontsize=15)
# plt.title('(c) RF', y=0.2, fontsize=15)
plt.xlabel('假阳率', fontsize=15)
plt.ylabel('真阳率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()

# GBDT
skplt.metrics.plot_roc(y_test, y_proba_GBDT, cmap='Set1', text_fontsize=15)
# plt.title('(e) GBDT', y=0.2, fontsize=15)
plt.xlabel('假阳率', fontsize=15)
plt.ylabel('真阳率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()

# LGBM
skplt.metrics.plot_roc(y_test, y_proba_LGBM, cmap='Set1', text_fontsize=15)
# plt.title('(f) LightGBM', y=0.2, fontsize=15)
plt.xlabel('假阳率', fontsize=15)
plt.ylabel('真阳率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)

legend_font = {"family" : "Times New Roman"}
plt.legend(fontsize=15, frameon=True, prop=legend_font)
plt.show()