import pandas as pd

# Load data and labels
t_acc_x_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_x_train.txt", header=None, sep='\s+')
t_acc_y_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_y_train.txt", header=None, sep='\s+')
t_acc_z_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_z_train.txt", header=None, sep='\s+')
labels = pd.read_csv("UCI+HAR+Dataset/train/y_train.txt", header=None, sep='\s+')

# Sum all the variances over all rows. We want the variance over time, not the variance between participants
# so we have to take axis 1 to get the variance over the rows. These are then summed and then compared.
# We don't need to normalize for the comparison, since they would all be normalized by the same factor (amount of rows).
sum_x = (sum(t_acc_x_train.var(axis=1)), 'x')
sum_y = (sum(t_acc_y_train.var(axis=1)), 'y')
sum_z = (sum(t_acc_z_train.var(axis=1)), 'z')
most_t_acc = max(sum_x, sum_y, sum_z)

print("The axis with the most variance is " + most_t_acc[1]
      + " with a summed variance over the rows of " + str(most_t_acc[0]) + ".")

# Load the dataset corresponding with our calculated axis.
# In this dataset each row represents 2.56s of measurements (50 Hz), where the values recorded are the acceleration
# along a certain axis. There are 7532 rows and 128 columns. Of each row, the first half is equal to the last half of the previous row
# and the last half is equal to the first half of the next row (overlap).
body_acc_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_acc_" + most_t_acc[1] + "_train.txt", header=None, sep='\s+')
print(body_acc_train.info())

# Take the middle of each window, as this is most likely to correspond to the given label. Other possibilities that
# Unoverlap the data include taking the first or last half, or just taking every second row.
unoverlapped = body_acc_train.iloc[:,32:96]

# Merge with labels
unoverlapped_labels = unoverlapped.merge(labels, how='outer', left_index=True, right_index=True)
print(unoverlapped_labels.info())


# get values as numpy array and flatten.
signal = unoverlapped.values.flatten()
print(signal)
