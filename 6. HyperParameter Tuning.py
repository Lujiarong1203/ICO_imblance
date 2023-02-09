import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
# plt.rc('font',family='Times New Roman')

config={"font.family": 'serif',
        "font.size": 15,
        "mathtext.fontset": 'stix',
        "font.serif": ['SimSun']}

plt.rcParams.update(config)



# plt.rcParams['font.sans-serif']=['SimHei']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

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

# # Parameter tuning/Each time a parameter is tuned, update the parameter corresponding to other_params to the optimal value


# 1-learning_rate
cv_params= {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = LGBMClassifier(random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the learning_rate validation_curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='learning_rate',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label="训练集得分")

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label="验证集得分")

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(a) learning_rate', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()


# 2-n_estimators
cv_params= {'n_estimators': range(60, 160, 10)}
model = LGBMClassifier(learning_rate=0.4, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the n_estimators validation_curve
param_range_1=range(60, 160, 10)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

print(train_scores_1, '\n', train_mean_1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10,label='验证集得分')

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(b) n_estimators', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()



# 3-max_depth
cv_params= {'max_depth': range(10, 20, 1)}
model = LGBMClassifier(learning_rate=0.4, n_estimators=120, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(10, 20, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='max_depth',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='验证集得分')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(c) max_depth', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()

# 4-num_leaves
cv_params= {'num_leaves': range(20, 30, 1)}
model = LGBMClassifier(learning_rate=0.4, n_estimators=130, max_depth=17, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(20, 30, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='num_leaves',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='验证集得分')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(d) num_leaves', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()

# 5-min_child_sample
cv_params= {'min_child_samples': range(7, 17, 1)}
model = LGBMClassifier(learning_rate=0.4, n_estimators=120, max_depth=17, num_leaves=24, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1=range(7, 17, 1)
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='min_child_samples',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='验证集得分')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(e) min_child_sample', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()


# 6-min_child_weight
cv_params= {'min_child_weight': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]}
model = LGBMClassifier(learning_rate=0.4, n_estimators=120, max_depth=17, num_leaves=24, min_child_samples=12, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='min_child_weight',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='验证集得分')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) min_child_weight', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()


# 7-colsample_bytree
cv_params= {'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
model = LGBMClassifier(learning_rate=0.4, n_estimators=120, max_depth=17, num_leaves=24, min_child_samples=12, min_child_weight=0.001, random_state=random_seed)
optimized_LGBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring="accuracy", cv=kf, verbose=1, n_jobs=-1)
optimized_LGBM.fit(x, y)
print('The best value of the parameter：{0}'.format(optimized_LGBM.best_params_))
print('Best model score:{0}'.format(optimized_LGBM.best_score_))

# Draw the max_depth validation curve
param_range_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x,
                                             y=y,
                                             param_name='colsample_bytree',
                                             param_range=param_range_1,
                                             cv=kf, scoring="accuracy", n_jobs=-1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="orange", linewidth=3.0,
         marker='v', markersize=10, label='训练集得分')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="orange")

plt.plot(param_range_1, test_mean_1, color="forestgreen", linewidth=3.0,
         marker='X', markersize=10, label='验证集得分')

plt.fill_between(param_range_1, test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="forestgreen")

plt.grid(visible=True, axis='y')
# plt.xscale('log')
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('参数值', fontsize=15)
plt.ylabel('准确率', fontsize=15)
plt.xticks(fontproperties='Times New Roman', fontsize=15)
plt.yticks(fontproperties='Times New Roman', fontsize=15)
plt.title('(f) colsample_bytree', y=-0.25, fontproperties='Times New Roman', fontsize=15)
plt.ylim([0.98, 1.0])
plt.tight_layout()
plt.show()

# 最佳参数：model = LGBMClassifier(learning_rate=0.1, n_estimators=100, max_depth=17, num_leaves=31, min_child_samples=15, min_child_weight=0.001, colsample_bytree=0.2, random_state=random_seed)

score_data_0=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(LGBMClassifier(random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data_0 = score_data_0.append(pd.DataFrame({'Before tuning': [score.mean()]}), ignore_index=True)
# print(score_data_0)

score_data_1=pd.DataFrame()
scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
for sco in scoring:
    score = cross_val_score(LGBMClassifier(learning_rate=0.4,
                                           n_estimators=120,
                                           max_depth=17,
                                           num_leaves=24,
                                           min_child_samples=12,
                                           min_child_weight=0.001,
                                           colsample_bytree=1,
                                           random_state=random_seed), x, y, cv=kf, scoring=sco)
    score_data_1 = score_data_1.append(pd.DataFrame({'After tuning': [score.mean()]}), ignore_index=True)
# print(score_data_1)

score_com=pd.concat([score_data_0, score_data_1], axis=1)
score_COM=score_com.rename(index={0:'accuracy', 1:'precision', 2:'recall', 3:'f1', 4:'roc_auc'})

print(score_COM.T)

