import pandas as pd
import numpy as np
from scipy import stats as st
import csv
import time
from sklearn import svm
import random


totalfilecount = 200
testfraction = 0.2

if __name__ == "__main__":
    testblocks = random.sample(range(1, totalfilecount), int(totalfilecount*testfraction))
    trainblocks = list(set(range(1,totalfilecount)) - set(testblocks))

    svm_classifier = svm.SVC(gamma='scale')

    all_features = None
    all_labels = None
    for block in trainblocks:
        data = pd.read_csv("AF_Filtered_Data/Data{}.csv".format(block), sep=' ', header=0).values
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

    svm_classifier.fit(all_features, all_labels)

    all_features = None
    all_labels = None
    for block in testblocks:
        data = pd.read_csv("AF_Filtered_Data/Data{}.csv".format(block), sep=' ', header=0).values
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


    res = svm_classifier.predict(all_features)
    # print(all_labels)
    # print(sum([(a + b) % 2 for (a,b) in zip(res, all_labels) ]))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(res)):
        if all_labels[i] != 0 and res[i] != 0:
            tp += 1
        elif all_labels[i] == 0 and res[i] == 0:
            tn += 1
        elif all_labels[i] != 0 and res[i] == 0:
            fn += 1
        elif all_labels[i] == 0 and res[i] != 0:
            fp += 1

    print("Confusion Matrix:")
    print("{:4} | {:4}\n------------\n{:4} | {:4}".format(tp, fn, fp, tn))
    print()
    print("Recall AF: {}".format(tp / (tp + fn)))
    print("Precision AF: {}".format(tp / (tp + fp)))