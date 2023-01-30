import pandas as pd

data = pd.read_csv(r"/Users/sunyong/Desktop/bishe/data/creditcard/creditcard.csv", header=0)
print(data)
print(data["Class"].value_counts())
