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
    train_features = pd.read_csv("trainfeatures_new.csv", sep=' ', header=0).values
    train_labels = pd.read_csv("trainlabels_new.csv", sep=' ', header=0).values
    test_features = pd.read_csv("testfeatures_new.csv", sep=' ', header=0).values
    test_labels = pd.read_csv("testlabels_new.csv", sep=' ', header=0).values
    test_labels = test_labels.reshape((test_labels.shape[0],))
    train_labels = train_labels.reshape((train_labels.shape[0],))
    
   
    yaf_feat = train_features[train_labels == 1.]
    fact = int(len(train_features) / len(yaf_feat)) - 1
    print(fact)
    for i in range(fact):
        train_features = np.concatenate((train_features, yaf_feat))
        train_labels = np.concatenate((train_labels, np.ones(len(yaf_feat))))

    nnclassifier = Sequential()
    nnclassifier.add(InputLayer(input_shape=(train_features.shape[1],)))
    nnclassifier.add(Dense(train_features.shape[1], activation='sigmoid'))
    nnclassifier.add(Dense(1, activation='sigmoid'))
    nnclassifier.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])



    nnclassifier.fit(train_features, train_labels, epochs=2)
    
    

    res = nnclassifier.predict(test_features)
    res = res.reshape((res.shape[0],))
    # print(all_labels)
    # print(sum([(a + b) % 2 for (a,b) in zip(res, all_labels) ]))
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(res)):
        if test_labels[i] == 1 and res[i] > 0.5:
            tp += 1
        elif test_labels[i] == 0 and res[i] <= 0.5:
            tn += 1
        elif test_labels[i] == 1 and res[i] <= 0.5:
            fn += 1
        elif test_labels[i] == 0 and res[i] > 0.5:
            fp += 1
        else:
            print("wtf: {}, {}".format(test_labels[i], res[i]))

    print("Confusion Matrix:")
    print("{:4} | {:4}\n------------\n{:4} | {:4}".format(tp, fn, fp, tn))
    print()
    print("{} wrong out of {}, ratio: {:.4e}".format(fp + fn, tp + fp + tn + fn,
                                                     (fp + fn) / (tp + fp + tn + fn)))
    print
    print("Recall AF: {}".format(tp / (tp + fn)))
    print("Precision AF: {}".format(tp / (tp + fp)))
