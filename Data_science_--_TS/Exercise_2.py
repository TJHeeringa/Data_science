import pandas as pd
import matplotlib.pyplot as plt


train_label_series = pd.read_csv("UCI+HAR+Dataset/train/Y_train.txt", header=None, sep='\s+')
print(train_label_series.info())
print(train_label_series.head())
print(train_label_series.columns)

plt.figure()
train_label_series.hist()
x = train_label_series[0].value_counts()

plt.figure()
plt.bar(x.index, x.values)

plt.show()
