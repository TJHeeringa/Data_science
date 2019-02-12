import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
