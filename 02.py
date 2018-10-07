import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import randint
from sklearn.impute import SimpleImputer

# 打印所有的列
# pd.set_option('max_columns', None)
# # 调节宽度使得所有列一行输出，不截断
# pd.set_option('display.width', 200)

housing = pd.read_csv('housing.csv')
#
# # 查看模型信息
print(housing.head(), end='\n\n')
# print(housing.info(), end='\n\n')
# print(housing["ocean_proximity"].value_counts(), end='\n\n')
# print(housing.describe(), end='\n\n')
# # bin  每个直方图里面柱形的数量   figszie图片大小（长，宽）
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()
#
# # 通过random_state指定随机种子，同个种子保证每次运行训练集是一样的
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# print(train_set.head(), end='\n\n')
# housing["median_income"].hist()
# plt.show()
#
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
# # inplace=True 在原来的DaTaFrame上修改
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
# housing["income_cat"].hist()
# plt.show()
#
# # n_splits 分割的组数
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    # print(strat_train_set.head(), end='\n\n')
    # print(strat_test_set.head(), end='\n\n')
#
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
#
housing = strat_train_set.copy()
# # alpha每个点的透明度，多个点叠加透明的增加，可方便看出点的集中区域
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
#

# s 每个点的大小，此处根据population列的大小决定点的大小
# c 指定根据某列值的大小显示不同的颜色
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True,
#     sharex=False)
# plt.legend()
# plt.show()

# 参数相关性查看，皮尔逊相关系数和散点图
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1)
# #设置坐标轴范围  xmin, xmax, ymin, ymax
# plt.axis([0, 16, 0, 550000])
# plt.show()

# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
# housing["population_per_household"] = housing["population"] / housing["households"]

housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
# sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
# print(sample_incomplete_rows)

# imputer = SimpleImputer(strategy="median")
# 删除非数值属性
housing_num = housing.drop('ocean_proximity', axis=1)
# imputer.fit(housing_num)
# print(imputer.statistics_)
# 返回Numpy 数组
# X = imputer.transform(housing_num)
# 转换成pandas  可通过index指定为原来的下标，此次可以不用指定
# housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
# print(housing_tr.loc[sample_incomplete_rows.index.values])
# housing_cat = housing[['ocean_proximity']]
# print(housing_cat.head(10))

# 文本转化为数字： 1)单个数字2)独热向量
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
# print(housing_cat_encoded[:10])
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot.toarray())

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# sklearn流水线处理，避免不断调用fix,transform等冗余代码
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])
housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)

# 预测
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))
#
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
# lin_mae = mean_absolute_error(housing_labels, housing_predictions)
# print(lin_mae)
#
# tree_reg = DecisionTreeRegressor(random_state=42)
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# # rmse=0 过拟合
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)
#
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# # 10次的平均rmse
# print(tree_rmse_scores.mean())
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# print(lin_rmse_scores.mean())
# #
# forest_reg = RandomForestRegressor(random_state=42)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                                 scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# print(pd.Series(forest_rmse_scores).describe())
#
# svr_score = cross_val_score(SVR(kernel="linear"), housing_prepared, housing_labels,
#                             scoring="neg_mean_squared_error", cv=10)
# svr_rmse_scores = np.sqrt(-svr_score)
# print(svr_rmse_scores.mean())

# 模型微调
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True, n_jobs=8)
grid_search.fit(housing_prepared, housing_labels)
# print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

# forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42,n_jobs=8)
rnd_search.fit(housing_prepared, housing_labels)
# print(rnd_search.best_params_, np.sqrt(-rnd_search.best_score_))

# 结果最好的预测器
final_model = rnd_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
