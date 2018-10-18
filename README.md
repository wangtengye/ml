# ml

机器学习代码，关于《Hands-On Machine Learning with Scikit-Learn and TensorFlow》

## 9.23

本周主要学习了Python的相关知识。

IDE：PyCharm

## 9.30

- 基础类库学习：NumPy，Pandas,sklearn
- 一些概念：
  -   监督学习：数据有标签
  -   无监督学习：数据无标签
  -   半监督学习：数据少量有标签，大量无标签
  -   强化学习：从经验学习，如AlphaGo
  -   遗传算法：适者生存

- sklearn算法选择：

![](https://raw.githubusercontent.com/wangtengye/image/master/20181018122442.png)

## 10.7

- 机器学习类型：
    -    监督，非监督，半监督和强化学习
    -    在线学习，批量学习
    -    基于实例学习，基于模型学习
- 步骤
    -    研究数据
    -    选择模型
    -    用训练数据进行训练
    -    使用模型对新案例进行预测
- 面临挑战
    -    训练数据量不足
    -    没有代表性的训练数据
    -    低质量数据
    -    不相关的特征      
    -    过拟合训练数据      
    -    欠拟合训练数据
- 函数
    -    pandas.read_csv()
    -    DataFrame. head()  info()  describe()  corr()（判断属性的相关系数）
    -    Series. value_counts()  hist()
- 训练集，测试集划分
    -    随机划分：train_test_split
    -    分层：StratifiedShuffleSplit
- 数据清洗
    -    去掉空值对应的行  DataFrame.dropna()
    -    去掉整个属性      DataFrame.drop()
    -    赋值            DataFrame.fillna()  Imputer.fit() transform()  fit_transform
- 属性转换   
  -    LabelEncoder  
  -    OneHotEncoder

- 特征缩放
  - 归一化  MinMaxScaler
  - 标准化  StandardScaler

- 训练模型​    
  - 回归:
    - 线性LinearRegression() 
    - 决策树 DecisionTreeRegressor  
    - 随机森林 RandomForestRegressor

- 交叉验证
  - K折交叉验证  cross_val_score()
  - 交叉验证试验所有可能超参数值的组合 GridSearchCV()
  - 随机搜索 RandomizedSearchCV

- sklearn. pipeline的调用

  ```
  设Pileline p由A,B,C组成
  则    p.fit()=A.fit()+A.transform()+B.fit()+B.transform()+C.fit() 
  	  p. transform()=A.transform()+B.transform()+C.transform()
  	  p. fit_transform= A.fit()+A.transform()+B.fit()+B.transform()+C.fit()+C.transform()
  
  ```

- 一次完整的预测过程

  - 由其他值预测median_house_value

    ![](https://raw.githubusercontent.com/wangtengye/image/master/20181018124647.png)

  - 数据预处理：填充缺失值，按median_income分层取样，文本属性独热编码，增加组合属性rooms_per_household，population_per_household，特征缩放

  - 采用 均方根误差（RMSE）来评价和 K折交叉验证，采取K=10

    | 回归方法 | 10此平均RMSE       |
    | -------- | ------------------ |
    | 线性回归 | 69052.46136345083  |
    | 决策树   | 71407.68766037929  |
    | 随机森林 | 52583.724074       |
    | SVR回归  | 111809.84009600841 |

    大多数街区的 median_housing_values 位于 120000到 265000 美元之间，从数据上看随机森林模型最好

  - 对随机森林模型微调进行最终预测

    | 网格搜索微调 | 47768.153251015996 |
    | ------------ | ------------------ |
    | 随机搜索微调 | 46910.92117024934  |

     

    可见不同微调方法得到的结果差不多

-  感慨

    虽然本次预测展示数据很少，但是前期做了不少尝试，包括查看属性相关联系性，预处理，模型的验证，评价的标准，花费了挺多时间进行了一些了解，虽然最后的结果算不上特别好，但是对于从头开始的我来说，算是机器学习新手路上的一大步。

## 10.14

- 二分类器

  - 分类器 SGDClassifier  RandomForestClassifier

  - 性能评估：

    - 交叉验证 cross_val_score ： 有偏差的数据集时不是一个好的性能度量指标，例如有两个类别，A类别只占10%，B类别占了90%，对于每次预测都返回B，至少也有90%的正确率。

    - 混淆矩阵：

      -   涉及函数：cross_val_predict  confusion_matrix，precision_score, recall_score， f1_score，precision_recall_curve，roc_curve，roc_auc_score，    predict_proba()

      -  输出组成（这里针对二分类器而言）

      | TN(真反例) | FP(假正例) |
      | ---------- | ---------- |
      | FN(假反例) | TP(真正例) |

      - 分数判别：
        - 准确率=   $\frac{TP}{FP+TP}$                                                     
        - 召回率=    $ \frac{TP}{FN+TP} $  
        - F1 值：准确率和召回率的调和平均
        - 不能同时提高准确率和召回率    
        - 根据图像判别：准确率/召回率曲线（或者叫 PR曲线）   ROC曲线
        -  规则:优先使用 PR 曲线当正例很少，或者当你关注假正例多于假反例的时候。其他情况使用 ROC 曲线。

- 多类分类
  - 二分类器处理多类分类问题一些方法：
    - 创建一个可以将图片分成 10 类（从 0 到 9）的系统的一个方法是：训练10分类器，每一个对应一个数字（探测器 0，探测器 1，探测器 2，以此类推）。然后当你想对某张图片进行分类的时候，让每一个分类器对这个图片进行分类，选出决策分数最高的那个分类器。这叫做“一对所有”（OvA）策略（也被叫做“一对其他”）。
    - 另一个策略是对每一对数字都训练一个二分类器：一个分类器用来处理数字 0 和数字 1，一个用来处理数字 0 和数字 2，一个用来处理数字 1 和 2，以此类推。这叫做“一对一”（OvO）策略。如果有 N 个类。你需要训练 N*(N-1)/2 个分类器。
  -  一些函数：decision_function，OneVsRestClassifier，predict_proba()
  -  plt.matshow() 图像呈现混淆矩阵  数字越大，图像越深. 

- 线性回归的不同梯度下降算法

  m:训练样本个数，n特征个数

  | 算法           | m很大 | 支持外存储 | n很大 | 超参数个数 | 特征缩放 | sklearn          |
  | -------------- | ----- | ---------- | ----- | ---------- | -------- | ---------------- |
  | 正态方程       | 快    | 不支持     | 慢    | 0          | 不需要   | LinearRegression |
  | 批量梯度下降   | 慢    | 不支持     | 快    | 2          | 需要     | 没有             |
  | 随机梯度下降   | 快    | 支持       | 快    | >=2        | 需要     | SGDRegressor     |
  | 小批量梯度下降 | 快    | 支持       | 快    | 》=2       | 需要     | 没有             |

- 多项式回归

  - 使用线性回归，先转换特征：PolynomialFeatures  输入$X$，可生成$X^{2}$

  - 判断过拟合还是欠拟合

    | 训练集分数 | 交叉验证分数 | 判断   |
    | ---------- | ------------ | ------ |
    | 好         | 不好         | 过拟合 |
    | 不好       | 不好         | 欠拟合 |

  - 正则化

    -  Ridge 回归  L2范数
    -  Lasso 回归  L1范数
    - 弹性网络（ElasticNet）

- Logistic 回归 Softmax 回归  回归用于分类

- SVM：线性或者非线性的分类，回归，甚至异常值检测

  - 分类

    - 线性：LinearSVC(C=1, loss="hinge") SVC(kernel="linear", C=1)

    - 非线性：

        - 增加特征PolynomialFeatures

        - 多项式核SVC(kernel="poly")

        - 增加相似特征(RBF)
        - 高斯 RBF 核  SVC(kernel="rbf")

    - 选择：一般来说，先尝试线性核函数（记住 LinearSVC 比 SVC(kernel="linear") 要快得多），尤其是当训练集很大或者有大量的特征的情况下。如果训练集不太大，也可以尝试高斯径向基核（Gaussian RBFKernel），它在大多数情况下都很有效。如果你有空闲的时间和计算能力，可以使用交叉验证和网格搜索来试验其他的核函数，特别是有专门用于你的训练集数据结构的核函数。

    - 比较  m:训练样本个数，n特征个数

    | 算法          | 时间复杂度                            | 支持外存储 | 特征缩放 | 核技巧 |
    | ------------- | ------------------------------------- | ---------- | -------- | ------ |
    | LinearSVC     | $O(m\times n)$                        | 不支持     | 需要     | 不支持 |
    | SGDClassifier | $O(m\times n)$                        | 支持       | 需要     | 不支持 |
    | SVC           | $O(m^{2}\times n)$~$O(m^{3}\times n)$ | 不支持     | 需要     | 支持   |

  - 回归

    - 线性 LinearSVR
    - 非线性 核化  SVR(kernel="poly")

 

- 一个分类预测样例    

  简介：用手写数字（0~9）图像训练，训练模型能够自动识别图片为哪个数字。

  - 读取数据

    数据由70000张图片组成，每个图片${28*28}$像素，故有每张图片有784个特征。则模型为${70000*784}$的二维矩阵

  - 预测

    | 分类模型                             | 准确度     |
    | ------------------------------------ | ---------- |
    | RandomForestClassifier               | 0.94359718 |
    | SGDClassifier                        | 0.88173226 |
    | KNeighborsClassifier  （耗时相对长） | 0.96939847 |

  - 误差的一些分析

    ![](https://raw.githubusercontent.com/wangtengye/image/master/20181018103251.png)

    错误率混淆矩阵图形化，亮度越亮，数值越大，可见5和3之间，9和7之间容易被误判为对方。



## 10.21

-  决策树
     -  多功能机器学习算法：分类，回归，多输出任务
          -  export_graphviz()可视化（下面仅针对分类而言）
               -  samples：训练样本实例
               -  value：每个类别的样本个数
               -  class：类别
               -  gini：纯度
       -  使用CART算法，产生二叉树
       -  白盒模型 
       -  predict_proba 属于每个类的概率
       -  复杂度 m:训练样本个数，n特征个数   预测：logm  训练：nmlogm
       -  分类目标使纯度或熵最小，回归目标使MSE最小
       -  存在问题：不稳定，对微小变化非常敏感，
-  集成学习
  -  VotingClassifier
     -  硬投票
     -   软投票
  -  Bagging 有放回采样  BaggingClassifier BaggingRegressor
  -  Pasting 无放回采样   BaggingClassifier(bootstrap=False)
  -  总体上Bagging会导致更好的模型 
  -  Bagging采样 Out-of-Bag评价  
  -  采样特征，采样实例
  -  随机森林  RandomForestClassifier  RandomForestRegressor
  -  极端随机树  ExtraTreesClassifier  ExtraTreesRegressor
  -  Adaboost 分类器：对之前分类结果不对的训练实例多加关注
  -  梯度提升：拟合残差   GradientBoostingRegressor
  -  Stacking：把所有分类器的结果当作输入构建模型输出
  -  XGBoost
-  降维
   -  维度越高，过拟合的风险越大
   -  主要方法 
      -  ·投影
      -  流形学习
      -  主成分分析（PCA）

## 参考资料

[Python教程](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)

[Python教程](https://github.com/yidao620c/python3-cookbook)

[Python教程](https://github.com/yidao620c/python3-cookbook)

[NumPy教程](https://juejin.im/post/5a76d2c56fb9a063557d8357)

[Pandas](http://pandas.pydata.org/pandas-docs/stable/10min.html)

[机器学习](https://morvanzhou.github.io)

[回归评价指标](https://morvanzhou.github.io)

[Hands-On Machine Learning with Scikit-Learn and
TensorFlow源码](https://github.com/ageron/handson-ml)

[Hands-On Machine Learning with Scikit-Learn and
TensorFlow中译本](https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF)

[matplotlib插值方法](https://matplotlib.org/gallery/images_contours_and_fields/interpolation_methods.html?highlight=interpolation)

[混淆矩阵](https://blog.csdn.net/m0_38061927/article/details/77198990)

[Python numpy 中 keepdims 的含义](https://blog.csdn.net/u012560212/article/details/78393836)

[KNN](https://zhuanlan.zhihu.com/p/22345658)

[NumPy随机函数](https://blog.csdn.net/u012149181/article/details/78913167)

[集成方法](http://sklearn.apachecn.org/cn/stable/modules/ensemble.html)