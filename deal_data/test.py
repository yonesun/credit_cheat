# import pandas as pd
# from sklearn.datasets import load_iris
# iris = load_iris() # 导入鸢尾花数据集
# df = pd.DataFrame(data=iris.data, columns=[i.replace(' ', '_')for i in iris.feature_names]) # 特征转DataFrame
# df['target'] = iris.target # 添加目标值
# df = df[df.target.isin([0, 1 ])] # 取目标值中的0,1类型的数据，用来做二分类算法
#
# # 分割数据集，用来训练模型
# x_train = df.drop('target', axis=1)
# y_train = df['target']
#
# # 加载LGBM模型（不需要训练）
# from lightgbm import LGBMClassifier
# model = LGBMClassifier()
#
# # fit后保存为pmml文件
# from sklearn2pmml import PMMLPipeline, sklearn2pmml
# pipeline = PMMLPipeline([('classifier', model)])
# pipeline.fit(x_train, y_train)
# sklearn2pmml(pipeline, 'mo
# del_fit_to_pmml.pmml', with_repr=True)
import faulthandler

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier


def testNP():
    print(np.arange(49))


def testModel():
    faulthandler.enable()
    X = np.random.rand(50, 3)
    y = np.ones((50, 1)).ravel()
    model = LGBMClassifier()
    model.fit(X, y)
    print(model.predict(np.random.rand(10, 3)))

# testModel()