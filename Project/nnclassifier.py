import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.engine import InputLayer
from tensorflow.python.keras.layers import Dense
from sklearn import svm
import random


totalfilecount = 805
testfraction = 0.2

if __name__ == "__main__":
    testblocks = random.sample(range(1, totalfilecount), int(totalfilecount*testfraction))
    trainblocks = list(set(range(1,totalfilecount)) - set(testblocks))



    all_features = None
    all_labels = None
    for block in trainblocks:
        data = pd.read_csv("AF_Filtered_Data2/Data{}.csv".format(block), sep=' ', header=0).values
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

    # yaf_feat = all_features[all_labels == 1]
    #
    # fact = int(len(all_features) / len(yaf_feat)) - 1
    #
    # for i in range(fact):
    #     all_features = np.concatenate(all_features, yaf_feat)
    #     all_labels = np.concatenate(all_labels, np.ones(len(yaf_feat)))


    nnclassifier = Sequential()
    nnclassifier.add(InputLayer(input_shape=(all_features.shape[1],)))
    nnclassifier.add(Dense(all_features.shape[1], activation='sigmoid'))
    nnclassifier.add(Dense(1, activation='sigmoid'))
    nnclassifier.compile(loss='mse', optimizer='adam', metrics=['mse'])



    nnclassifier.fit(all_features, all_labels, epochs=1)


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


    res = nnclassifier.predict(all_features)
    res = res.reshape((res.shape[0],))
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
    print("{} wrong out of {}, ratio: {:.4e}".format(fp + fn, tp + fp + tn + fn,
                                                     (fp + fn) / (tp + fp + tn + fn)))
    print
    print("Recall AF: {}".format(tp / (tp + fn)))
    print("Precision AF: {}".format(tp / (tp + fp)))