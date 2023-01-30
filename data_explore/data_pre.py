import numpy as np
import pandas as pd
from data_ana import dataAna, standard
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

data = dataAna()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = standard(X)
X.columns = ['Amount', 'Time', 'V1', 'V10', 'V11', 'V12', 'V13', 'V14',
             'V15', 'V16', 'V17', 'V18', 'V19', 'V2', 'V20', 'V21', 'V24', 'V26',
             'V27', 'V28', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']
allDataX = pd.read_csv("fakeMake20230129_104202.csv", header=0).iloc[:283823, :]
allDataX.columns = ['Amount', 'Time', 'V1', 'V10', 'V11', 'V12', 'V13', 'V14',
                    'V15', 'V16', 'V17', 'V18', 'V19', 'V2', 'V20', 'V21', 'V24', 'V26',
                    'V27', 'V28', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9']
allDataY = pd.DataFrame(np.ones([284315 - 492, 1]))

# 此处输入数据处理完毕，size=284315*2
dataX = pd.concat([X, allDataX], ignore_index=True)
dataY = pd.concat([y, allDataY], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=1)
X_train_ini, X_test_ini, y_train_ini, y_test_ini = train_test_split(X_train, y_train, test_size=0.3, random_state=1)
models = [SVC(), LogisticRegression(), RandomForestClassifier(), LGBMClassifier(learning_rate=0.1, n_estimators=100)]
X_test_ini_res = pd.DataFrame()

"""
测试
# model = LGBMClassifier(learning_rate=0.1, n_estimators=100)
# model.fit(X_train_ini, y_train_ini.values.ravel())
# print(model.predict(X_test_ini))
"""

for model in models:
    clf = model.fit(X_train_ini, y_train_ini.values.ravel())
    predict_result = pd.DataFrame(clf.predict(X_test_ini))
    print(predict_result)
    X_test_ini_res = pd.concat([X_test_ini_res, predict_result], axis=1)
print(X_test_ini_res)
