import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, KFold
from imblearn import over_sampling,under_sampling,combine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold, cross_validate, RandomizedSearchCV

# import scikitplot as skplt
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data= pd.read_csv("data/data_3.csv")
print(data.shape)
print(Counter(data['fraud']))

random_seed=1234

x=data.drop('fraud', axis=1)
y=data.fraud
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_seed)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# View the label distribution in the training and test sets
print('train_set:', Counter(y_train), '\n', 'test_set:', Counter(y_test))

# # Save the test set data and training set data for later parameter adjustment and evaluation
data_train=pd.concat([x_train, y_train], axis=1)
data_test=pd.concat([x_test, y_test], axis=1)
print(data_train.shape, data_test.shape)
# data_train.to_csv(path_or_buf=r'data/data_train.csv', index=None)
# data_test.to_csv(path_or_buf=r'data/data_test.csv', index=None)

# 混合采样SMOTE+Tomek Links
X_resampled, y_resampled = combine.SMOTETomek(random_state=random_seed).fit_resample(x_train, y_train)
print(sorted(Counter(y_resampled).items()))


# # standardized
# 标准化采样后的训练集
mm = MinMaxScaler()
X_resampled_std = pd.DataFrame(mm.fit_transform(X_resampled))
X_resampled_std.columns = X_resampled.columns
data_train_std=pd.concat([X_resampled_std, y_resampled], axis=1)
print(data_train_std.head(5), data_train_std.shape)

# 标准化测试集
mm = MinMaxScaler()
X_test_std = pd.DataFrame(mm.fit_transform(x_test))
X_test_std.columns = x_test.columns
y_test.reset_index(drop=True, inplace=True)
# print(y_test.head(10))
data_test_std=pd.concat([X_test_std, y_test], axis=1)
print(data_test_std.head(5), data_test_std.shape)

data_train_std.to_csv(path_or_buf=r'data/data_train.csv', index=None)
data_test_std.to_csv(path_or_buf=r'data/data_test.csv', index=None)
