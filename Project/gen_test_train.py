import pandas as pd
import numpy as np

totalfilecount = 805
testfraction = 0.2

if __name__ == "__main__":
    split = int(totalfilecount* (1.-testfraction))
    trainblocks = list(range(1,split))
    testblocks = list(range(split,totalfilecount))

    all_features = None
    all_labels = None
    for block in trainblocks:
        data = pd.read_csv("AF_Filtered_Data_test/Data{}.csv".format(block), sep=' ', header=0).values
        if data.shape[0] == 0:
            continue
        valids = np.isfinite(data[:,-1].reshape((data.shape[0],)))
        data = data[valids]
        features = data[:,range(data.shape[1]-1)]
        labels = data[:,-1]
        if all_features is None:
            all_features = features
        else:
            all_features = np.concatenate((all_features, features))

        if all_labels is None:
            all_labels = labels
        else:
            all_labels = np.concatenate((all_labels, labels))

    np.savetxt("trainfeatures.csv", all_features, delimiter=' ', header=' samples "o - mean" "o - var" "o - max" "o - min" "o - spread" "o - mad" "o - kurtosis" "o - skew" "o - median" "o - first quantile" "o - last quantile" "o - sum" "d - mean" "d - var" "d - max" "d - min" "d - spread" "d - mad" "d - kurtosis" "d - skew" "d - median" "d - first quantile" "d - last quantile" "d - sum" "rm - mean" "rm - var" "rm - max" "rm - min" "rm - spread" "rm - mad" "rm - kurtosis" "rm - skew" "rm - median" "rm - first quantile" "rm - last quantile"')
    np.savetxt("trainlabels.csv", all_labels.reshape((all_labels.shape[0], 1)), delimiter=' ', header='label')

    all_features = None
    all_labels = None
    for block in testblocks:
        data = pd.read_csv("AF_Filtered_Data_test/Data{}.csv".format(block), sep=' ', header=0).values
        if data.shape[0] == 0:
            continue
        valids = np.isfinite(data[:, -1].reshape((data.shape[0],)))
        data = data[valids]
        features = data[:, range(data.shape[1] - 1)]
        labels = data[:, -1]
        if all_features is None:
            all_features = features
        else:
            all_features = np.concatenate((all_features, features))

        if all_labels is None:
            all_labels = labels
        else:
            all_labels = np.concatenate((all_labels, labels))

    np.savetxt("testfeatures.csv", all_features, delimiter=' ',
               header=' samples "o - mean" "o - var" "o - max" "o - min" "o - spread" "o - mad" "o - kurtosis" "o - skew" "o - median" "o - first quantile" "o - last quantile" "o - sum" "d - mean" "d - var" "d - max" "d - min" "d - spread" "d - mad" "d - kurtosis" "d - skew" "d - median" "d - first quantile" "d - last quantile" "d - sum" "rm - mean" "rm - var" "rm - max" "rm - min" "rm - spread" "rm - mad" "rm - kurtosis" "rm - skew" "rm - median" "rm - first quantile" "rm - last quantile"')
    np.savetxt("testlabels.csv", all_labels.reshape((all_labels.shape[0], 1)), delimiter=' ', header='label')