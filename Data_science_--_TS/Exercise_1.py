import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------
# Exercise 1a
# ------------------------------------------
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


# ------------------------------------------
# Exercise 1b
# ------------------------------------------
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


# ------------------------------------------
# Exercise 1c
# ------------------------------------------
train_data = pd.read_csv("UCI+HAR+Dataset/train/X_train.txt", header=None, sep='\s+')
train_label = pd.read_csv("UCI+HAR+Dataset/train/Y_train.txt", header=None, sep='\s+')

merged_dataframe = train_data.merge(train_label, how='outer', left_index=True, right_index=True)

print(" --- data --- ")
print(train_data.info())
print(" --- label --- ")
print(train_label.info())
print(" --- merged --- ")
print(merged_dataframe.info())

dataframes = [merged_dataframe[merged_dataframe['0_y'] == i] for i in range(6)]

for k in range(1, 11):
    plt.figure()
    print(["{}".format(i) for i in range(6)])
    for dataframe in dataframes:
        sns.kdeplot(dataframe.loc[:, k], legend=False)
    plt.legend(["{}".format(i) for i in range(1, 6)])
    plt.title("Plot for feature {}".format(k))
plt.show()
