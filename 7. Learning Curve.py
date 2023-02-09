import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
import xgboost
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
# K-fold
kf=KFold(n_splits=5, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in kf.split(x, y):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# LR
LR=LogisticRegression(random_state=random_seed)

# SVM
SVM=SVC(random_state=random_seed)

# RF
RF=RandomForestClassifier(random_state=random_seed)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)

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

# 1_LR
skplt.estimators.plot_learning_curve(LR, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) LR', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 2_SVM
skplt.estimators.plot_learning_curve(SVM, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) SVM', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 3_Ada
skplt.estimators.plot_learning_curve(Ada, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) Adaboost', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 4_GBDT
skplt.estimators.plot_learning_curve(GBDT, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(d) GBDT', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 5_RF
skplt.estimators.plot_learning_curve(RF, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(e) RF', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()

# 6_LGBM
skplt.estimators.plot_learning_curve(LGBM, x, y, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('样本量', fontsize=15)
plt.ylabel('得分', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) LightGBM', y=-0.2, fontproperties='Times New Roman', fontsize=15)
plt.tight_layout()
plt.show()







