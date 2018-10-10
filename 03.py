from sklearn.datasets import fetch_mldata

# 加载数据
mnist = fetch_mldata('MNIST original')
print(mnist)

X, y = mnist["data"], mnist["target"]
# (70000, 784)  70000 张图片，每张图片有 784 个特征，784=28*28像素
print(X.shape)

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
# # cmap颜色控制 interpolation 插值方法，不同方法渲染图像不同
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
#            interpolation="nearest")
# # 不展示坐标轴
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
import numpy as np

shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# ------------------------以下都是针对二分类器，只有真假-------------------------------------
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=5, random_state=42, n_jobs=8)
# sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))
#
# # 交叉验证5的识别率
from sklearn.model_selection import cross_val_score
#
# print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#
# from sklearn.base import BaseEstimator
#
#
# class Never5Classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         pass
#
#     def predict(self, X):
#         return np.zeros((len(X), 1), dtype=bool)


# 对所有的预测值返回False也有90%的准确率
# ，因为在数据集中非5的数字占90%，总是返回False只有5的预测才是失败的
# 对于分类器而言，这种验证方法不好
# never_5_clf = Never5Classifier()
# print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
#
from sklearn.model_selection import cross_val_predict
#
# # 每次取3份，每次取出一份当测试集
# y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
from sklearn.metrics import confusion_matrix
#
# # 混淆矩阵  组成样子：
# # TN(真反例)  FP(假正例)
# # FN(假反例)  TP(真正例)
# print(confusion_matrix(y_train_5, y_train_pred))
#
# from sklearn.metrics import precision_score, recall_score
#
# # 准确率  0.8764121102575689
# # 声明某张图片是 5 的时候，它只有 87.6% 的可能性是正确的，其他的5可能是4，3等等
# print(precision_score(y_train_5, y_train_pred))
# # 召回率 0.7155506364139458
# # 只检测出“是 5”类图片当中的 71.6%,其他的5判别错误为3，4等等
# print(recall_score(y_train_5, y_train_pred))
# from sklearn.metrics import f1_score
#
# # F1 值是准确率和召回率的调和平均
# print(f1_score(y_train_5, y_train_pred))
#
# y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
#                              method="decision_function")
# from sklearn.metrics import precision_recall_curve
#
# precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#
#
# def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
#     # "b--" b代表蓝色 --代表虚线
#     plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
#     plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
#     plt.xlabel("Threshold", fontsize=16)
#     plt.legend(loc="upper left", fontsize=16)
#     plt.ylim([0, 1])
#
#
# plt.figure(figsize=(8, 4))
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.xlim([-700000, 700000])
# plt.show()
#
# # 提高阈值  准确率增加，召回率降低
# y_train_pred_90 = (y_scores > 70000)
# print(precision_score(y_train_5, y_train_pred_90))
# print(recall_score(y_train_5, y_train_pred_90))
#
#
# def plot_precision_vs_recall(precisions, recalls):
#     plt.plot(recalls, precisions, "b-", linewidth=2)
#     plt.xlabel("Recall", fontsize=16)
#     plt.ylabel("Precision", fontsize=16)
#     plt.axis([0, 1, 0, 1])
#
#
# # 准确率/召回率曲线（或者叫 PR）
# plt.figure(figsize=(8, 6))
# plot_precision_vs_recall(precisions, recalls)
# plt.show()
#
# from sklearn.metrics import roc_curve
#
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
#
#
# def plot_roc_curve(fpr, tpr, label=None):
#     plt.plot(fpr, tpr, linewidth=2, label=label)
#     # [0,1],[0,1]  ->[x1,x2][y1,y2]
#     # 随机的分类器生成的ROC
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.axis([0, 1, 0, 1])
#     plt.xlabel('False Positive Rate', fontsize=16)
#     plt.ylabel('True Positive Rate', fontsize=16)
#
#
# # 受试者工作特征（ROC）曲线
# plt.figure(figsize=(8, 6))
# plot_roc_curve(fpr, tpr)
# plt.show()
#
# # 测量ROC曲线下的面积（AUC）。一个完美的分类器的 ROC AUC 等于 1，而一个纯随机分类器的 ROC AUC 等于 0.5
# from sklearn.metrics import roc_auc_score
#
# print(roc_auc_score(y_train_5, y_scores))
#
# # 规则:优先使用 PR 曲线当正例很少，或者当你关注假正例多于假反例的时候。
# # 其他情况使用 ROC 曲线。所以此处的判别应该使用PR曲线
#
#
from sklearn.ensemble import RandomForestClassifier

#
forest_clf = RandomForestClassifier(random_state=42, n_jobs=8)
# y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
#                                     method="predict_proba")
# print(y_probas_forest)
#
# # 把正例的概率当分数
# y_scores_forest = y_probas_forest[:, 1]
# fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
#
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="lower right", fontsize=16)
# plt.show()
# print(roc_auc_score(y_train_5, y_scores_forest))
#
# # 该模型的准确率和召回率
# y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
# print(precision_score(y_train_5, y_train_pred_forest))
# print(recall_score(y_train_5, y_train_pred_forest))

# ------------------------以下为多类分类-------------------------------------
# 二分类器处理多类分类问题一些方法：
# 创建一个可以将图片分成 10 类（从 0 到 9）的系统的一个方法是：训练10个二分类
# 器，每一个对应一个数字（探测器 0，探测器 1，探测器 2，以此类推）。然后当你想对某张
# 图片进行分类的时候，让每一个分类器对这个图片进行分类，选出决策分数最高的那个分类
# 器。这叫做“一对所有”（OvA）策略（也被叫做“一对其他”）。
# 另一个策略是对每一对数字都训练一个二分类器：一个分类器用来处理数字 0 和数字 1，一
# 个用来处理数字 0 和数字 2，一个用来处理数字 1 和 2，以此类推。这叫做“一对一”（OvO）
# 策略。如果有 N 个类。你需要训练 N*(N-1)/2 个分类器。

# 使用OvA策略
# sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))
#
# # 验证是OvA
# some_digit_scores = sgd_clf.decision_function([some_digit])
# print(some_digit_scores)
# print(np.argmax(some_digit_scores))
# print(sgd_clf.classes_)
# print(sgd_clf.classes_[5])

# 强制使用OVO
from sklearn.multiclass import OneVsOneClassifier

# ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))
# 分类器数量  N=10  10*(10-1)/2=45
# print(len(ovo_clf.estimators_))
# 没使用策略，天生支持多分类
# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
print(cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))
# 图片预测为该类的概率   ，结果显示有90%被预测为5
# print(forest_clf.predict_proba([some_digit]))
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1))
from sklearn.preprocessing import StandardScaler
#
# # 输入标准化后精度提高
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))
#
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# 图像呈现混淆矩阵  数字越大，图像越深
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
# # keepdims  保持维数
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]

knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
# verbose 日志输出控制   n_jobs=-1利用所有核心
# grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)
# print(grid_search.best_score_)
# knn_clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# y_pred = knn_clf.predict(X_test)
# 0.9714
# print(accuracy_score(y_test, y_pred))
# [0.97365527 0.97169858 0.96979547]  elapsed: 18.0min finished 耗时大概18分钟
print(cross_val_score(knn_clf, X_train, y_train, cv=3, scoring="accuracy", verbose=3, n_jobs=-1))
