import pandas as pd

df = pd.read_csv("UCI+HAR+Dataset/train/X_train.txt", header=None, sep='\s+')
print(df.info())
print(df.head())
print(df.columns)
x = df.loc[:, [1, 2, 3, 4, 5]]
#
print("hi")
print(x.info())
print(" -- mean --")
print(x.mean())
print(" -- var --")
print(x.var())
print(" -- med --")
print(x.median())
print(" -- min --")
print(x.min())
print(" -- max --")
print(x.max())
