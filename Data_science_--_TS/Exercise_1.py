import pandas as pd

print(" -----------------------------------------------------------")
print(" --- A --- ")
print(" -----------------------------------------------------------")
df = pd.read_csv("UCI+HAR+Dataset/train/X_train.txt", header=None, sep='\s+')
print(df.info())
print(df.head())
print(df.columns)

print("This dataset has 561 columns and 7352 rows.")
print("Each columns has time and frequency domain features.")
print("The rows are measurements at certain timestamps.")
print("Each participant performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) for a certain amount of time")
print("The data recorded for each participant has been concatonated.")

print(" -----------------------------------------------------------")
print(" --- B --- ")
print(" -----------------------------------------------------------")
x = df.loc[:, [1, 2, 3, 4, 5]]
print(" -- info -- ")

print(x.info())
print(" -- mean -- ")
print(x.mean())
print(" -- var -- ")
print(x.var())
print(" -- med -- ")
print(x.median())
print(" -- min -- ")
print(x.min())
print(" -- max -- ")
print(x.max())

