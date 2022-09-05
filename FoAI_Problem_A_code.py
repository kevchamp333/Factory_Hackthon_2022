''' 기본 모듈 및 시각화 모듈 '''
from IPython.display import display, HTML
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' 데이터 전처리 모듈 '''
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

''' 결과 평가용 모듈 '''
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, r2_score

''' 기타 optional'''
# =============================================================================
# import warnings, itertools
# warnings.filterwarnings(action='ignore')
# pd.set_option('display.max_columns', None)
# =============================================================================

'''황우영'''
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor

'''정용곤'''
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor, plot_importance
from sklearn.neural_network import MLPRegressor

# =============================================================================
# 데이터 전처리 (NA 지우고, 의미없는 column 제거)
# =============================================================================

os.chdir(r'C:\Users\Woo Young Hwang\Desktop\SPS\외부 활동\대회\경진대회\산업인공지능 해커톤\데이터')
raw_data = pd.read_excel('A기업_데이터 (비식별화).xlsx', sheet_name = 'data')

raw_data.iloc[:,1:101].info()

data = raw_data.dropna(axis=1)
data.drop(['Work Time', 'Serial No.'], axis=1, inplace=True)

data.info()

drop_list = []

for i in range(len(data.columns)):
    # 필요하면 그림 출력하기
    # print(data.iloc[:,i].describe())
    # plt.plot(data.iloc[:,i])
    # plt.title(data.columns[i])
    # plt.legend(data.columns[i], loc='best')
    # plt.show()
    if data.iloc[:,i].dtype == 'O':
        pass
    elif (data.iloc[:,i].dtype != 'O') & (data.iloc[:,i].describe()['std'] == 0):
        print(data.columns[i])
        drop_list.append(data.columns[i])
    else:
        pass

drop_list

data.drop(drop_list, axis=1, inplace=True)
data.info()

# =============================================================================
# feature, target 정의 & 더미변수 생성
# =============================================================================

features = data.drop(['G', 'T1', 'T2', 'W_R', 'W_L'], axis=1)
target_G = data['G']
target_T1 = data['T1']
target_T2 = data['T2']
target_W_R = data['W_R']
target_W_L = data['W_L']
target_mean = data[['G', 'T1', 'T2', 'W_R', 'W_L']].mean(axis = 1)

# target 변수의 distribution 확인
ax1 = sns.distplot(data['G'])
ax2 = sns.distplot(data['T1'])
ax3 = sns.distplot(data['T2'])
ax4 = sns.distplot(data['W_R'])
ax5 = sns.distplot(data['W_L'])
plt.legend(labels = ['G', 'T1', 'T2', 'W_R', 'W_L'])
plt.show()

display(features.head())
display(target_G.head())
display(target_T1.head())
display(target_T2.head())
display(target_W_R.head())
display(target_W_L.head())

data['Part Name']
data['Part Name'].value_counts()

temp = features.drop(labels='Part Name', axis=1)
dummy = pd.get_dummies(data=data['Part Name'], prefix='Part Name')
# dummy = pd.get_dummies(data=data['Part Name'], prefix='Part Name', drop_first=True)
dummy

features = pd.concat(objs=[dummy, temp], axis=1)
column_list = features.columns
features.info()

# =============================================================================
# 변수간 corr 체크 (데이터 대분류를 중심으로) (1) 공정데이터, (2) Bowl 공정데이터, (3) Shank 공정데이터
# =============================================================================

corr_1 = features.iloc[:, :11].corr()
plt.figure(figsize=(len(corr_1.columns),len(corr_1.columns)))
sns.heatmap(corr_1, annot=True, fmt='.1g')

corr_2 = features.iloc[:, 11:31].corr()
plt.figure(figsize=(len(corr_2.columns),len(corr_2.columns)))
sns.heatmap(corr_2, annot=True, fmt='.1g')

corr_3 = features.iloc[:, 31:].corr()
plt.figure(figsize=(len(corr_3.columns),len(corr_3.columns)))
sns.heatmap(corr_3, annot=True, fmt='.1g')

# =============================================================================
# 학습데이터 정규화 1) Standardization
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(features, target_mean, test_size=0.3, random_state=0)

scaler = StandardScaler()
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=column_list)
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=column_list)

# =============================================================================
# ML model (LR, KNN-R, DT, RF, SVM, GBM, LightGBM)
# =============================================================================

tree = DecisionTreeRegressor().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
knn = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
ranfo = RandomForestRegressor(n_estimators=300, max_depth=None).fit(X_train, y_train)
gb_reg = GradientBoostingRegressor(loss = "ls", learning_rate = 0.1, n_estimators = 500, criterion = "mse",
                                  max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, verbose = 1, random_state=0).fit(X_train, y_train)
lgbm_reg = LGBMRegressor(task = 'train', boosting_type = 'gbdt', objective = 'regression', metric = 'l2', num_iterations = 10000).fit(X_train, y_train)
reg_mlp = MLPRegressor(activation='identity', alpha=0.001, hidden_layer_sizes=(25), max_iter=1000, solver='adam', verbose = True, random_state = 0).fit(X_train, y_train)


# X_test의 Decision Tree Regression 결과
tree_prediction = tree.predict(X_test)
# X_test의 Linear Regression 결과
lr_prediction = lr.predict(X_test)
# X_test의 KNN Regression 결과
knn_prediction = knn.predict(X_test)
# X_test의 Random Forest Regression 결과
ranfo_prediction = ranfo.predict(X_test)
# X_test의 GradientBoostingRegressor 결과
gb_reg_prediction = gb_reg.predict(X_test)
# X_test의 LGBM Regression 결과
lgbm_reg_prediction = lgbm_reg.predict(X_test)
# X_test의 MLP Regression 결과
reg_mlp_prediction = reg_mlp.predict(X_test)

# =============================================================================
# # ensemble model
# =============================================================================
def get_stacking():
    #base models
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('SVR', SVR()))
    level0.append(('RFR', RandomForestRegressor(n_estimators=100, max_depth=None)))
    level0.append(('GBR', GradientBoostingRegressor(n_estimators = 100, criterion = "mse", random_state=0)))
    level0.append(('LightGBM', LGBMRegressor(task = 'train', boosting_type = 'gbdt', objective = 'regression', metric = 'l2', n_estimators = 100)))
    #meta learner model
    level1 = LinearRegression()
    #stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model

def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['SVR'] = SVR()
    models['RFR'] = RandomForestRegressor(n_estimators=100, max_depth=None)
    models['GBR'] = GradientBoostingRegressor(n_estimators = 100, criterion = "mse", random_state=0)
    models['LightGBM'] = LGBMRegressor(task = 'train', boosting_type = 'gbdt', objective = 'regression', metric = 'l2', n_estimators = 100)
    models['stacking'] = get_stacking()
    return models

# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# define dataset
X = X_train
y = y_train

models = get_models()

results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model, X, y)
    results.append(np.sqrt(scores*-1))
    names.append(name)
    print('>%s %.3f (%.3f)' % (name, np.mean(np.sqrt(scores*-1)), np.std(np.sqrt(scores*-1))))     #RMSE

plt.boxplot(results, labels=names, showmeans=True)
plt.show()

#fit stacking model
model = get_stacking().fit(X_train, y_train)

# X_test의 Stacking Regression 결과
stacking_prediction = model.predict(X_test)

# 예측 결과와 실제 결과 비교를 위해 plot 그림
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel('data points')
plt.ylabel('target')
plt.plot(stacking_prediction, 'ro-')
plt.plot(np.array(y_test), 'o-')
plt.legend(["ML Model", "y_test"])
plt.show()

# RMSE of test model
RMSE_tree = mean_squared_error(stacking_prediction, np.array(y_test))**0.5
RMSE_tree

# r2 of test model
r2 = r2_score(stacking_prediction, np.array(y_test)) 
r2
# =============================================================================
# Multi Layer Perceptron Regressor
# =============================================================================

reg_mlp = MLPRegressor(activation='identity', alpha=0.001, hidden_layer_sizes=(25),
                       max_iter=1000, solver='adam', verbose = True, random_state = 0)

reg_mlp.fit(X_train, y_train)

plt.figure(figsize=(20,10))
train_loss_values = reg_mlp.loss_curve_
plt.plot(train_loss_values,label='Train Loss')
plt.legend(fontsize=20)
plt.title("Learning Curve of trained MLP Regressor", fontsize=18)
plt.show()

MLP_pred = reg_mlp.predict(X_test)

# 예측 값과 실제 값 간의 차이(오차) 계산
# Mean Absolute Error (MAE)
mlp_mae = mean_absolute_error(y_test, MLP_pred)

# Mean squared Error (MSE)
mlp_mse = mean_squared_error(y_test, MLP_pred)

# Root Mean Squared Error (RMAE)
mlp_rmae = (np.sqrt(mean_absolute_error(y_test, MLP_pred)))

# Root Mean Squared Error (RMSE)
mlp_rmse = (np.sqrt(mean_squared_error(y_test, MLP_pred)))

print("Testing performance")
print('mlp_MAE: {:.4f}'.format(mlp_mae))
print('mlp_MSE: {:.4f}'.format(mlp_mse))
print('mlp_RMAE: {:.4f}'.format(mlp_rmae))
print('mlp_RMSE: {:.4f}'.format(mlp_rmse))


# =============================================================================
# 변수 중요도 plotting
# =============================================================================


'''decision tree feature importance'''
# decision tree 쓰면 필요 --> 변수 중요도
importances = tree.feature_importances_

# 내림차순으로 정렬하기 위한 index
index = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title('DT Feature Importances')
plt.bar(range(features.shape[1]), importances[index], align='center')
plt.xticks(range(features.shape[1]), features.columns[index], rotation=90)
plt.xlim([-1, features.shape[1]])
plt.show()


'''Gradient Boosting Regressor feature importance'''
# feature importance 산출 (feature importance 값이 높을수록 feature 중요도가 높다고 해석할 수 있음)
feature_importance = gb_reg.feature_importances_

# feature_importance를 높은 순서로 정렬
sorted_idx = np.argsort(feature_importance)[::-1]

fig = plt.figure(figsize=(8, 6))
plt.bar(range(features.shape[1]), feature_importance[sorted_idx], align='center')
plt.xticks(range(features.shape[1]), np.array(features.columns)[sorted_idx], rotation=90)
plt.xlim([-1, features.shape[1]])
plt.title('GBM Feature Importance')
plt.show()


'''LightGBM Regressor feature importance'''
# feature importance 산출 (feature importance 값이 높을수록 feature 중요도가 높다고 해석할 수 있음)
lgbm_reg_feature_importance = lgbm_reg.feature_importances_

# feature_importance를 높은 순서로 정렬
sorted_idx = np.argsort(lgbm_reg_feature_importance)[::-1]

fig = plt.figure(figsize=(8, 6))
plt.bar(range(features.shape[1]), lgbm_reg_feature_importance[sorted_idx], align='center')
plt.xticks(range(features.shape[1]), np.array(features.columns)[sorted_idx], rotation=90)
plt.xlim([-1, features.shape[1]])
plt.title('LightGBM Feature Importance')
plt.show()

# 다른 버전 (plot_importance( )를 이용하여 feature 중요도 시각화)
fig, ax = plt.subplots(figsize=(12, 6))
plot_importance(lgbm_reg, ax=ax)

