from collections import Counter
import functools
from heapq import *

import numpy as np
import pandas as pd
import random
import time

from multiprocessing.pool import Pool



# Simple distance function between 2 points
def d(x1,x2):
    return abs(x1-x2)


# Calculate DTW of 2 time series in a matrix of 2x(2*bound+3) (can be done in just an array of 2*bound+3)
# The bound is how much the indices of 2 points may differ
def dtw_low_space(a1, a2, bound, d=d):
    assert(len(a1) == len(a2))
    mid = bound + 1
    m = np.full((2,2*bound+3), np.infty)
    m[0,mid] = d(a1[0], a2[0])
    for i in range(1,len(a1)):
        k = i % 2
        l = 1-k
        for j in range(max(mid-i, 1), min(mid+(len(a1)-i-1), mid+bound)+1):
            m[k, j] = min(m[k, j - 1], m[l, j], m[l, j + 1]) + d(a1[i], a2[i-mid+j])
    return m[(len(a1)-1) % 2, mid]


#implementation in even less space
def dtw_no_space(a1, a2, bound, d=d):
    assert(len(a1) == len(a2))
    mid = bound + 1
    m = np.full((2*bound+3,), np.infty)
    m[mid] = d(a1[0], a2[0])
    for i in range(1,len(a1)):
        for j in range(max(mid-i, 1), min(mid+(len(a1)-i-1), mid+bound)+1):
            m[j] = min(m[j - 1], m[j], m[j + 1]) + d(a1[i], a2[i-mid+j])
    return m[mid]


# If you have multiple time series to compare, the result is just the sum
# euclidean distance does not apply here, so no pythagorean stuff
def multi_dtw_low_space(tss1, tss2, bound, d=d):
    assert(len(tss1) == len(tss2))
    total = 0
    for i in range(len(tss1)):
        total += dtw_low_space(tss1[i], tss2[i], bound, d)
    return total



'''
Base class the implements a KNN with distance function DTW, where the data is two-dimensional
In the case of our dataset, 9 time series of length 128
Complexity of train: O(1)
Complexity of predict: O(bound*l*m) where l is the length of a series and m is the size of the trainset
Complexity of predict_many: O(n*predict)
'''
class DTWKNN2D:

    # k is how many neighbours it looks for, proc is how many threads may be used in calculation
    def __init__(self, k=5, bound=50, proc=1):
        self.k = k
        self.proc = proc
        self.bound = bound

    def train(self, tsss, labels):
        # tsss is list of arrays where the array is a 9 by width combination of all signal data for a given frame
        assert(len(tsss) == len(labels))
        self.tsss = np.array(tsss)
        self.labels = np.array(labels)


    def predict(self, test):

        # Prepare function to be used in the python map function, which must be a function that only gets 1 argument
        comparefunc = functools.partial(multi_dtw_low_space, tss2=test, bound=self.bound)

        # From the inside out:
        # map: calculate distance to every point in our dataset
        # zip: combine with the labels in our dataset, so now we have a list of (dist, label) tuples
        # sorted: sort on distance, smalles first
        # np.array()[:self.k, 1]: make numpy array and take only the labels of the first k elements
        # astype(int): cast to ints for the counter
        # Counter().most_common: count the elements and take the highest count
        return Counter(np.array(sorted(zip(map(comparefunc, self.tsss), self.labels)))[:self.k, 1].astype('int')).most_common(1)[0][0]


    # Wrapper that accepts a list of inputs and can multiprocess them
    def predict_many(self, testset, chunksize=1):
        p = Pool(processes=self.proc)
        res = p.map(self.predict, testset, chunksize=chunksize)
        return res



'''
    This next part splits the DTW function in iterations, so we can have intermediate results.
    We can apply dijkstra to these intermediate results so we calculate the smallest ones first
    after having found the k smallest ones, the rest needs not be calculated
    Sadly, the overhead of the heapq is way more than the early cutoff gains, with only some exceptions
    Complexity is at worst case the same as base class complexity times log (m)
'''

def multi_dtw_init(tss1, tss2, bound):
    assert (tss1.shape == tss2.shape)
    tssm = np.full((tss1.shape[0], 2*bound+3), np.infty)
    total = 0
    for i in range(len(tss1)):
        total += dtw_no_space_init(tssm[i], tss1[i], tss2[i])
    return total, tssm

def multi_dtw_iteration(tssm, i, tss1, tss2):
    total = 0
    for j in range(tssm.shape[0]):
        total += dtw_no_space_iter(tssm[j], i, tss1[j], tss2[j])
    return total

def dtw_no_space_init(m, a1, a2, d=d):
    assert(len(a1) == len(a2))
    mid = m.shape[0]//2
    m[mid] = d(a1[0], a2[0])
    return m[mid]

def dtw_no_space_iter(m, i, a1, a2, d=d):
    assert (len(a1) == len(a2))
    mid = m.shape[0]//2
    low = np.infty
    for j in range(max(mid - i, 1), min(mid + (len(a1) - i), m.shape[0]-1)):
        m[j] = min(m[j - 1], m[j], m[j + 1]) + d(a1[i], a2[i - mid + j])
        if m[j] < low:
            low = m[j]
    return low


class DKNN(DTWKNN2D):

    def predict(self, test):
        initfunc = functools.partial(multi_dtw_init, tss2=test, bound=50)
        q = [(t, m, 1, l, tss) for ((t, m), l, tss) in zip(map(initfunc, self.tsss), self.labels, self.tsss)]
        heapify(q)
        ctr = 0
        res = []
        while ctr < self.k:
            (t,m,i,l,tss) = q[0]
            if i >= tss.shape[1]:
                res.append(l[0])
                ctr += 1
                heappop(q)
                continue
            newt = multi_dtw_iteration(m, i, tss, tss2=test)
            if newt == np.infty:
                continue
            heapreplace(q, (newt, m, i+1, l, tss))
        return Counter(res).most_common(1)[0][0]




'''
    Different idea: We can determine an upper bound to the distance to each other series.
    For this we determine some beacons, for which the distance to all other series is known.
    Then for each pair of series, an upper bound is the sum of the distances of s1 to beacon to s2.
    With multiple beacons we can take the lowest upper bound.
    Then we can determine the kth upper bound and we know that at least k elements have a distance smaller than that
    So if a calculation exceeds this distance, we can stop.
    This methods is faster on this dataset. It performs about 1.5 times as fast
'''


def upperbound_multi_dtw_low_space(tss1, tss2, bound, upper, d=d):
    assert(len(tss1) == len(tss2))
    total = 0
    for i in range(len(tss1)):
        total += dtw_no_space(tss1[i], tss2[i], bound, d)
        if total > upper:
            total = np.infty
            #print("Aborted after iteration {}.".format(i))
            return total, i
    #print("not aborted")
    return total, 8


class BoundedKNN(DTWKNN2D):
    def __init__(self, k=5, bound=50, proc=1, beacons=20, verbose=False):
        DTWKNN2D.__init__(self, k=k, bound=bound, proc=proc)
        self.beacon_count = beacons
        self.verbose = verbose

    def train(self, tsss, labels):
        DTWKNN2D.train(self, tsss, labels)
        p = Pool(self.proc)
        self.beacon_ids = random.sample(range(len(tsss)), k=self.beacon_count)
        self.beacon_dists = np.full((self.beacon_count, len(tsss)), np.infty)
        for i, id in enumerate(self.beacon_ids):
            if self.verbose:
                print("starting beacon {}: {}".format(i, id))
            comparefunc = functools.partial(multi_dtw_low_space, tss2=self.tsss[id], bound=self.bound)
            self.beacon_dists[i] = np.array(p.map(comparefunc, self.tsss))

        if self.verbose:
            print("training complete")

    def predict(self, test):
        max_dist = np.full((len(self.tsss),), np.infty, dtype=float)

        for i, id in enumerate(self.beacon_ids):
            dist = multi_dtw_low_space(self.tsss[id], test, self.bound)
            # if self.verbose:
            #     print("distance to beacon {}: {}".format(id, dist))
            for j in range(len(self.tsss)):
                # if self.verbose:
                #     print("oldmax: {:8}, newmax:{:8}".format(max_dist[j], min(max_dist[j], dist + self.beacon_dists[i][j])))
                max_dist[j] = min(max_dist[j], dist + self.beacon_dists[i][j])
        upper_bound = nsmallest(self.k, max_dist)[-1]
        comparefunc = functools.partial(upperbound_multi_dtw_low_space, tss2=test, bound=self.bound, upper=upper_bound)
        res = np.array([[t,a] for (t,a) in map(comparefunc, self.tsss)])
        avg_abort = sum(res[:,1])/len(self.tsss)
        print("Avg abort: {}".format(avg_abort))
        return Counter(np.array(sorted(zip(res[:,0], self.labels)))[:self.k, 1].astype('int')).most_common(1)[0][0]




'''
    This one has the same idea as above, but then also tries to determine a lower bound.
    For every series of which the lower bound is higher than the lowest upper bound, we just throw out immediately
    Sadly we have not found a reliable lower bound calculation, so this produces invalid results
'''
class BeaKNN(DTWKNN2D):

    def __init__(self, k=5, bound=50, proc=1, beacons=20, verbose=False):
        DTWKNN2D.__init__(self, k=k, bound=bound, proc=proc)
        self.beacon_count = beacons
        self.verbose = verbose

    def train(self, tsss, labels):
        DTWKNN2D.train(self, tsss, labels)
        p = Pool(self.proc)
        self.beacon_ids = random.sample(range(len(tsss)), k=self.beacon_count)
        self.beacon_dists = np.full((self.beacon_count, len(tsss)), np.infty)
        for i, id in enumerate(self.beacon_ids):
            if self.verbose:
                print("starting beacon {}: {}".format(i, id))
            comparefunc = functools.partial(multi_dtw_low_space, tss2=self.tsss[id], bound=self.bound)
            self.beacon_dists[i] = np.array(p.map(comparefunc, self.tsss))

        if self.verbose:
            print("training complete")

    def predict(self, test):
        global abortctr
        minmax = np.zeros((len(self.tsss), 2))
        for i in range(len(self.tsss)):
            minmax[i][0] = 0
            minmax[i][1] = np.infty
        for i, id in enumerate(self.beacon_ids):
            dist = multi_dtw_low_space(self.tsss[id], test, self.bound)
            # if self.verbose:
                # print("distance to beacon {}: {}".format(id, dist))
            for j in range(len(self.tsss)):
                # if self.verbose:
                    # print("oldmin: {:8}, newmin:{:8}".format(minmax[j][0], max(minmax[j][0], abs(dist-self.beacon_dists[i][j]))))
                    # print("oldmax: {:8}, newmax:{:8}".format(minmax[j][1], min(minmax[j][1], dist + self.beacon_dists[i][j])))
                minmax[j][0] = max(minmax[j][0], abs(dist-self.beacon_dists[i][j]))
                minmax[j][1] = min(minmax[j][1], dist + self.beacon_dists[i][j])
        upper_bound = nsmallest(self.k, minmax[:,1])[-1]
        newtrain = []
        newlabels = []
        for i in range(len(self.tsss)):
            if minmax[i][0] < upper_bound:
                newtrain.append(self.tsss[i])
                newlabels.append(self.labels[i])
        if self.verbose:
            print("Elminated {} series!".format(len(self.tsss) - len(newtrain)))
        calculator = DTWKNN2D(k=self.k, bound=self.bound, proc=1)
        calculator.train(newtrain, newlabels)
        return calculator.predict(test)






if __name__ == '__main__':

    body_acc_x_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_acc_x_train.txt", header=None,
                                   sep='\s+').values
    body_acc_y_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_acc_y_train.txt", header=None,
                                   sep='\s+').values
    body_acc_z_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_acc_z_train.txt", header=None,
                                   sep='\s+').values
    body_gyro_x_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_gyro_x_train.txt", header=None,
                                    sep='\s+').values
    body_gyro_y_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_gyro_y_train.txt", header=None,
                                    sep='\s+').values
    body_gyro_z_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/body_gyro_z_train.txt", header=None,
                                    sep='\s+').values
    total_acc_x_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_x_train.txt", header=None,
                                    sep='\s+').values
    total_acc_y_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_y_train.txt", header=None,
                                    sep='\s+').values
    total_acc_z_train = pd.read_csv("UCI+HAR+Dataset/train/Inertial Signals/total_acc_z_train.txt", header=None,
                                    sep='\s+').values

    trainset = np.dstack((body_acc_x_train, body_acc_y_train, body_acc_z_train, body_gyro_x_train, body_gyro_y_train,
                          body_gyro_z_train, total_acc_x_train, total_acc_y_train, total_acc_z_train)).swapaxes(1, 2)

    trainlabels = pd.read_csv("UCI+HAR+Dataset/train/y_train.txt", header=None, sep='\s+').values

    body_acc_x_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_acc_x_test.txt", header=None,
                                  sep='\s+').values
    body_acc_y_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_acc_y_test.txt", header=None,
                                  sep='\s+').values
    body_acc_z_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_acc_z_test.txt", header=None,
                                  sep='\s+').values
    body_gyro_x_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_gyro_x_test.txt", header=None,
                                   sep='\s+').values
    body_gyro_y_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_gyro_y_test.txt", header=None,
                                   sep='\s+').values
    body_gyro_z_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/body_gyro_z_test.txt", header=None,
                                   sep='\s+').values
    total_acc_x_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/total_acc_x_test.txt", header=None,
                                   sep='\s+').values
    total_acc_y_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/total_acc_y_test.txt", header=None,
                                   sep='\s+').values
    total_acc_z_test = pd.read_csv("UCI+HAR+Dataset/test/Inertial Signals/total_acc_z_test.txt", header=None,
                                   sep='\s+').values


    testset = np.dstack((body_acc_x_test, body_acc_y_test, body_acc_z_test, body_gyro_x_test, body_gyro_y_test,
                         body_gyro_z_test, total_acc_x_test, total_acc_y_test, total_acc_z_test)).swapaxes(1, 2)

    testlabels = pd.read_csv("UCI+HAR+Dataset/test/y_test.txt", header=None, sep='\s+').values

    starttime = time.time()
    # model = DKNN(k=5, proc=1, bound=20)

    model = BoundedKNN(k=10, proc=10, bound=10, verbose=True, beacons=20)
    # model = BeaKNN(k=5, proc=6, bound=20, verbose=True, beacons=10)

    # model = DTWKNN2D(k=10, proc=1, bound=20)
    test_amount = 1000
    train_amount = 2000
    train_indices = random.sample(range(trainset.shape[0]), train_amount)
    test_indices = random.sample(range(testset.shape[0]), test_amount)
    model.train(trainset[train_indices], trainlabels[train_indices])
    traintime = (time.time() - starttime)
    print("training took {:.3f} seconds".format(traintime))
    # res = model.predict(testset[0])
    res = model.predict_many(testset[test_indices])
    # print(res)
    # print(testlabels[offset:offset + amount])
    testtime = (time.time() - starttime - traintime)
    print("testing took {:.3f} seconds".format(testtime))
    m = np.zeros((6,6), dtype=np.int)
    for (pred, act) in zip(res, testlabels[test_indices]):
        m[pred-1][act-1] += 1

    print(m)

    '''
    For training input size 2000 and test size 1000, the elements of which were chosen randomly:
    testing took 5871.916 seconds
    
    [[182  32  40   0   0   0]
     [  1 113   0   0   0   0]
     [  0   1 103   0   0   0]
     [  0   0   0 138  43   0]
     [  0   0   0  28 142   0]
     [  0   0   0   1   0 176]]

    '''



