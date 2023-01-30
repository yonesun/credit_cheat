import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from deal_data.gan import GAN
from sklearn.preprocessing import MinMaxScaler
matplotlib.use("TkAgg")

def standard(df):
    model = MinMaxScaler((-1,1))
    result = model.fit_transform(df)
    return pd.DataFrame(result)

def dataAna():
    data = pd.read_csv(r"/Users/sunyong/Desktop/bishe/data/creditcard/creditcard.csv", header=0)
    # figure = plt.figure()
    # sns.displot(data["Amount"], color="b")
    # plt.ylim(0, 4000)
    # plt.xlim(0, 8000)
    # plt.show()
    data_copy = data.copy()
    # print(data["Time"].describe())

    Amount = np.array(data["Amount"].apply(lambda x: x + 1))
    l, lambdaValue = stats.boxcox(Amount)
    print("lambda值为：", lambdaValue)
    data["Amount"] = stats.boxcox(Amount, lmbda=lambdaValue)
    # fig = plt.figure()
    # sns.distplot(data["Amount"], color="b")
    # plt.show()

    data["Time"] = pd.cut(data["Time"], bins=3600 * np.arange(49), include_lowest=True, labels=np.arange(1, 49))
    data["Time"] = data["Time"].astype(int)
    # print(data["Time"])

    # fraudTime = data[data["Class"] == 1]["Time"]
    # fraudAmount = data_copy[data_copy["Class"] == 1]["Amount"]
    # fig = plt.figure()
    # plt.scatter(fraudTime, fraudAmount)
    # plt.show()
    fraud = data[data["Class"] == 1]
    fraudTimeCount = fraud.groupby(["Time"])["Class"].count().rename("Count")

    # fig = plt.figure()
    # plt.scatter(fraudTimeCount.index, fraudTimeCount)
    # plt.xlabel("Time")
    # plt.ylabel("Count")
    # plt.show()
    #
    # corr = data.corr().abs()
    # sns.heatmap(corr, cmap="Blues")
    # plt.show()

    rubbishX = ["V22", "V23", "V25"]
    columns = data.columns.difference(rubbishX)
    # print(columns)
    data = data[columns]
    count = data["Class"].value_counts()
    # print(count)
    # fig = plt.figure()
    # plt.bar(count.index, count)
    # plt.xlim([0,1])
    # plt.show()
    data = data[['Amount', 'Time', 'V1', 'V10', 'V11', 'V12', 'V13', 'V14',
                 'V15', 'V16', 'V17', 'V18', 'V19', 'V2', 'V20', 'V21', 'V24', 'V26',
                 'V27', 'V28', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'Class']]
    return data


# data = dataAna()
# print(data.describe())
#
#
# fraud = data[data["Class"] == 1]
# fraud_X = fraud[fraud.columns.difference(["Class"])].reset_index(drop=True)
# fraud_X_standard = standard(fraud_X)

# print(fraud_X_standard.describe())

# GAN(fraud_X_standard)
