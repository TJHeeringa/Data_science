import pandas
import matplotlib.pyplot as plt

train_data_frame = pandas.read_fwf("UCI+HAR+Dataset/train/X_train.txt")
train_label_frame = pandas.read_fwf("UCI+HAR+Dataset/train/y_train.txt")
print(train_data_frame.info())
print(train_label_frame.info())

train_label_series = train_label_frame.iloc[:,0]
print(train_label_series)

plt.figure()
train_label_series.plot(kind="bar")

plt.figure()
train_label_series.hist()

plt.show()
