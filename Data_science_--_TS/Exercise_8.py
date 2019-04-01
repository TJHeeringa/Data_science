from collections import Counter
import functools
from heapq import *

import numpy as np
import pandas as pd
import sys

from multiprocessing.pool import Pool

# sys.setrecursionlimit(10000)
#
#
# class DTW:
#
#
#     def __init__(self, a1, a2):
#         self.infty = float('inf')
#         self.a1 = a1
#         self.a2 = a2
#         self.result = self.dtw()
#
#     def dtw(self):
#         # self.m = np.zeros((len(self.a1), len(self.a2)), dtype=float)
#         # self.m = self.m - 1.
#         self.m = np.full((len(self.a1), len(self.a2)), self.infty)
#         self.bound = 50 # self.LB_Keogh(max(20, abs(len(self.a1) - len(self.a2))))
#         # return self.solve(len(self.a1)-1, len(self.a2)-1)
#         # return self.dijkstra()
#         return self.iterative()
#
#     # thanks to http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
#     def LB_Keogh(self, r):
#         LB_sum = 0
#         s1 = self.a1
#         s2 = self.a2
#         for ind, i in enumerate(s1):
#
#             lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
#             upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
#
#             if i > upper_bound:
#                 LB_sum = LB_sum + (i - upper_bound) ** 2
#             elif i < lower_bound:
#                 LB_sum = LB_sum + (i - lower_bound) ** 2
#
#         return np.sqrt(LB_sum)
#
#     @staticmethod
#     def dist(n, m):
#         return abs(n-m)
#
#
#     def iterative(self):
#         self.bound += 1
#         self.m[0,0] = DTW.dist(self.a1[0], self.a2[0])
#         for i in range(1, self.m.shape[0]):
#             self.m[i, 0] = self.m[i - 1, 0] + DTW.dist(self.a1[i], self.a2[0])
#         for j in range(1, self.m.shape[1]):
#             self.m[0, j] = self.m[0, j - 1] + DTW.dist(self.a1[0], self.a2[j])
#         for i in range(1, self.m.shape[0]):
#             for j in range(max(1,i - self.bound//2), min(i + self.bound//2, self.m.shape[1])):
#                 self.m[i,j] = min(self.m[i - 1, j - 1], self.m[i - 1, j], self.m[i, j - 1]) + DTW.dist(self.a1[i], self.a2[j])
#
#         rows = self.m.shape[0] - 1
#         cols = self.m.shape[1] - 1
#         return self.m[rows, cols]
#
#
#     def dijkstra(self):
#         for i in range(self.m.shape[0]):
#             for j in range(self.m.shape[1]):
#                 self.m[i,j] = self.infty if abs(i-j) > self.bound else DTW.dist(self.a1[i], self.a2[j])
#
#
#         q = []
#         heappush(q, (self.m[0,0], (0,0)))
#
#         rows = self.m.shape[0]-1
#         cols = self.m.shape[1]-1
#
#         visited = np.full(self.m.shape, False)
#
#         res = -1
#
#         while (len(q) > 0):
#             element = heappop(q)
#
#             if visited[element[1]]:
#                 continue
#             visited[element[1]] = True
#
#             if element[0] == self.infty:
#                 continue
#
#             # if (element[1][0] < 50 and element[1][1] < 50):
#             # print(element)
#
#             if (element[1] == (rows,cols)):
#                 res = element[0]
#                 break
#
#             next = []
#             if element[1][0] < rows:
#                 next.append((element[1][0] + 1, element[1][1]))
#                 if element[1][1] < cols:
#                     next.append((element[1][0] + 1, element[1][1] + 1))
#
#             if element[1][1] < cols:
#                 next.append((element[1][0], element[1][1] + 1))
#
#             for n in next:
#                 if not visited[n] and self.m[n] != self.infty:
#                     heappush(q, (element[0] + self.m[n], n))
#
#         return res
#
#
#
#     def solve(self, i : int, j : int):
#         #if (DTW.dist(self.a1[i],self.a2[j]) > self.bound):
#         if (abs(i-j) > self.bound):
#             return self.infty
#
#         if self.m[i,j] >= 0:
#             return self.m[i,j]
#         if i == 0 and j == 0:
#             self.m[i,j] = DTW.dist(self.a1[i],self.a2[j])
#             return self.m[i,j]
#
#         if i == 0:
#             self.m[i,j] = DTW.dist(self.a1[i], self.a2[j]) + self.solve(i, j-1)
#             return self.m[i,j]
#         if j == 0:
#             self.m[i,j] = DTW.dist(self.a1[i], self.a2[j]) + self.solve(i-1, j)
#             return self.m[i,j]
#
#         self.m[i, j] = DTW.dist(self.a1[i], self.a2[j]) + min(self.solve(i-1, j),self.solve(i, j-1),self.solve(i-1, j-1))
#         return self.m[i,j]
#
# def dtw_iterative(a1,a2,m,bound):
#     rows = len(a1)
#     cols = len(a2)
#     bound += 1
#     m[0,0] = DTW.dist(a1[0], a2[0])
#     for i in range(1, rows):
#         m[i, 0] = m[i - 1, 0] + DTW.dist(a1[i], a2[0])
#     for j in range(1, cols):
#         m[0, j] = m[0, j - 1] + DTW.dist(a1[0], a2[j])
#     for i in range(1, rows):
#         for j in range(max(1,i - bound//2), min(i + bound//2, cols)):
#             m[i,j] = min(m[i - 1, j - 1], m[i - 1, j], m[i, j - 1]) + DTW.dist(a1[i], a2[j])
#
#
#     return m[rows-1, cols-1]
#
#
# class KNN:
#
#     def __init__(self, k, proc=1):
#         self.k = k
#         self.proc = proc
#         self.infty = float('inf')
#         self.bound = 50
#         self.m = np.full((256,256), np.infty, dtype=float)
#
#     def train(self, tsss, labels):
#         # tsss is list of arrays where the array is a 9 by width combination of all signal data for a given frame
#         assert(len(tsss) == len(labels))
#         self.tsss = np.array(tsss)
#         self.labels = np.array(labels)
#
#
#     def comparesingle(self, s1, s2, m):
#         # return DTW(s1,s2).result
#         return dtw_iterative(s1,s2, m, self.bound)
#
#     def compareall(self, tss1, tss2, m):
#         assert(len(tss1) == len(tss2))
#         return np.sqrt(sum([self.comparesingle(tss1[i], tss2[i], m)**2 for i in range(len(tss1))]))
#
#     def predict(self, test):
#         m = np.full((128,128), np.infty, dtype=float)
#         comparefunc = functools.partial(self.compareall, tss2=test, m=m)
#         return Counter(np.array(sorted(zip(map(comparefunc, self.tsss), self.labels)))[:self.k,1].astype('int')).most_common(1)[0][0]
#
#
#     def predictmany(self, testset):
#         p = Pool(processes=self.proc)
#         res = p.map(self.predict, testset, chunksize=1)
#         return res
#
#
#
#
# class DKNN:
#
#     def __init__(self, k, proc=1):
#         self.proc = proc
#         self.k = k
#         self.infty = float('inf')
#         self.bound = 50
#         self.m = np.full((256,256), np.infty, dtype=float)
#
#     def train(self, tsss, labels):
#         # tsss is list of arrays where the array is a 9 by width combination of all signal data for a given frame
#         assert(len(tsss) == len(labels))
#         self.tsss = np.array(tsss)
#         self.labels = np.array(labels)
#
#
#     def predict(self, test):
#         # print("starting inits")
#         initfunc = functools.partial(multi_dtw_init, tss2=test, bound=50)
#         q = [(t, m, 1, l, tss) for ((t, m), l, tss) in zip(map(initfunc, self.tsss), self.labels, self.tsss)]
#         # print("init done, building heap")
#         heapify(q)
#         ctr = 0
#         res = []
#         # print("starting loop")
#         while ctr < self.k:
#             (t,m,i,l,tss) = q[0]
#             if i >= tss.shape[1]:
#                 res.append(l[0])
#                 ctr += 1
#                 heappop(q)
#                 continue
#             newt = multi_dtw_iteration(m, i, tss, tss2=test)
#             if newt == np.infty:
#                 continue
#             heapreplace(q, (newt, m, i+1, l, tss))
#         return Counter(res).most_common(1)[0][0]
#
#
#     def predictmany(self, testset):
#         p = Pool(processes=self.proc)
#         res = p.map(self.predict, testset, chunksize=1)
#         return res
#
#

def d(x1,x2):
    return abs(x1-x2)


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

def multi_dtw_low_space(tss1, tss2, bound, d=d):
    assert(len(tss1) == len(tss2))
    total = 0
    for i in range(len(tss1)):
        total += dtw_low_space(tss1[i], tss2[i], bound, d) # ** 2
    return total



class DTWKNN2D:

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
        comparefunc = functools.partial(multi_dtw_low_space, tss2=test, bound=self.bound)
        return Counter(np.array(sorted(zip(map(comparefunc, self.tsss), self.labels)))[:self.k, 1].astype('int')).most_common(1)[0][0]


    def predict_many(self, testset, chunksize=1):
        p = Pool(processes=self.proc)
        res = p.map(self.predict, testset, chunksize=chunksize)
        return res



def multi_dtw_init(tss1, tss2, bound):
    assert (tss1.shape == tss2.shape)
    halfbound = (bound)//2
    tssm = np.full((tss1.shape[0], 2, 2*halfbound+3), np.infty)
    total = 0
    for i in range(len(tss1)):
        total += dtw_init(tss1[i], tss2[i], tssm[i])**2
    return total, tssm

def multi_dtw_iteration(tssm, i, tss1, tss2):
    total = 0
    for j in range(tssm.shape[0]):
        total += dtw_iteration(tssm[j], i, tss1[j], tss2[j]) ** 2
    return total

def dtw_init(a1,a2,m):
    assert(len(a1) == len(a2))
    mid = m.shape[1]//2
    m[0, mid] = d(a1[0], a2[0])
    return m[0, mid]


def dtw_iteration(m, i, a1, a2):
    mid = m.shape[1]//2
    k = i%2
    l = 1-k
    halfbound = min(i, len(a1)-i-1, mid-2)
    lower = mid-halfbound
    upper = mid+halfbound + 1
    for j in range(-halfbound, halfbound+1):
        m[k, mid+j] = min(m[l, mid + j - 1], m[l, mid + j], m[l, mid + j + 1]) + d(a1[i], a2[i+j])
    return min(m[k, lower:upper+1])


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


import random


# def upperbound_dtw_low_space(a1, a2, bound, d=d):
#     assert(len(a1) == len(a2))
#     mid = bound // 2
#     m = np.full((2,2*mid+3), np.infty)
#     m[0,mid] = d(a1[0], a2[0])
#     for i in range(1,len(a1)):
#         k = i % 2
#         l = 1-k
#         halfbound = min(i, len(a1)-i-1, mid-2)
#         for j in range(-halfbound, halfbound+1):
#             m[k, mid+j] = min(m[l, mid + j - 1], m[l, mid + j], m[l, mid + j + 1]) + d(a1[i], a2[i+j])
#     return m[(len(a1)-1) % 2, mid]

def upperbound_multi_dtw_low_space(tss1, tss2, bound, upper, d=d):
    assert(len(tss1) == len(tss2))
    total = 0
    for i in range(len(tss1)):
        total += dtw_low_space(tss1[i], tss2[i], bound, d)
        if total > upper:
            total = np.infty
            #print("Aborted after iteration {}.".format(i))
            return total, i
    #print("not aborted")
    return total, 8
#
# def multi_dtw_low_space_sq(tss1, tss2, bound, d=d):
#     assert(len(tss1) == len(tss2))
#     total = 0
#     for i in range(len(tss1)):
#         total += dtw_low_space(tss1[i], tss2[i], bound, d) ** 2
#     return total


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
# trainset = [e for e in zip((body_acc_x_train,body_acc_y_train,body_acc_z_train,body_gyro_x_train,body_gyro_y_train,
#                   body_gyro_z_train,total_acc_x_train,total_acc_y_train,total_acc_z_train))]

trainset = np.dstack((body_acc_x_train,body_acc_y_train,body_acc_z_train,body_gyro_x_train,body_gyro_y_train,
                  body_gyro_z_train,total_acc_x_train,total_acc_y_train,total_acc_z_train)).swapaxes(1,2)

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
# testset = [e for e in zip((body_acc_x_test,body_acc_y_test,body_acc_z_test,body_gyro_x_test,body_gyro_y_test,
#                   body_gyro_z_test,total_acc_x_test,total_acc_y_test,total_acc_z_test))]

testset = np.dstack((body_acc_x_test,body_acc_y_test,body_acc_z_test,body_gyro_x_test,body_gyro_y_test,
                  body_gyro_z_test,total_acc_x_test,total_acc_y_test,total_acc_z_test)).swapaxes(1,2)

testlabels = pd.read_csv("UCI+HAR+Dataset/test/y_test.txt", header=None, sep='\s+').values


import time
if __name__ == '__main__':
    starttime = time.time()
    model = BoundedKNN(k=10, proc=6, bound=10, verbose=True, beacons=20)
    # model = BeaKNN(k=5, proc=6, bound=20, verbose=True, beacons=10)

    # model = DTWKNN2D(k=10, proc=14, bound=20)
    amount = 100
    offset = 1610
    model.train(trainset[:1000], trainlabels[:1000])
    traintime = (time.time() - starttime)
    print("training took {:.3f} seconds".format(traintime))
    # res = model.predict(testset[0])
    res = model.predict_many(testset[offset:offset + amount])
    print(res)
    print(testlabels[offset:offset + amount])
    testtime = (time.time() - starttime - traintime)
    print("testing took {:.3f} seconds".format(testtime))
# a1 = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])
# a2 = np.array([2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,98,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9])

# a1 = np.array([i for i in range(0,256,1)])
# a2 = np.array([i for i in range(256,0,-1)])
#
# print(len(a1))
# print(len(a2))
# print(DTW(a1,a2).result)
