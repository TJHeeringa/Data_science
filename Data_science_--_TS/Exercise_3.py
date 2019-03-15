import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ACTIVITY_DICT = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}
if __name__ == '__main__':
    train_data = pd.read_csv("UCI+HAR+Dataset/train/X_train.txt", header=None, sep='\s+')
    train_label = pd.read_csv("UCI+HAR+Dataset/train/Y_train.txt", header=None, sep='\s+')

    merged_dataframe = train_data.merge(train_label, how='outer', left_index=True, right_index=True)

    print(" --- data --- ")
    print(train_data.info())
    print(" --- label --- ")
    print(train_label.info())
    print(" --- merged --- ")
    print(merged_dataframe.info())

    dataframes = [merged_dataframe[merged_dataframe['0_y'] == i] for i in range(1, 7)]

    for k in range(1, 11):
        plt.figure()
        for dataframe in dataframes:
            sns.kdeplot(dataframe.loc[:, k], legend=False)
        plt.legend(["{}".format(ACTIVITY_DICT[i]) for i in range(1, 7)])
        plt.title("Plot for feature {}".format(k))
    plt.show()

    print("The distributions plotted are Gaussians; either a single or a sum of a couple.")
    print("Distinction on label is quite hard to do base on a single or couple of plots.")
    print("For example feature 1 has different heights, but similar mean and spread for each of the activities")
    print("This is a poor measure, since variaties in height occur naturally.")
    print("Looking at the plots of feature 4 and feature 5, we are hover able to detect the")
    print("difference between passive and active activities. The peaks are clearly separated.")
