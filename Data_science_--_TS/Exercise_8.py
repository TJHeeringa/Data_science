from collections import Counter
import functools
import numpy as np
import pandas as pd
import sys

from multiprocessing.pool import Pool

sys.setrecursionlimit(10000)


class DTW:


    def __init__(self, a1, a2):
        self.infty = float('inf')
        self.a1 = a1
        self.a2 = a2
        self.result = self.dtw()

    def dtw(self):
        self.m = np.zeros((len(self.a1), len(self.a2)), dtype=float)
        self.m = self.m - 1.
        self.bound = 2 # self.LB_Keogh(max(20, abs(len(self.a1) - len(self.a2))))
        return self.solve(len(self.a1)-1, len(self.a2)-1)

    # thanks to http://alexminnaar.com/time-series-classification-and-clustering-with-python.html
    def LB_Keogh(self, r):
        LB_sum = 0
        s1 = self.a1
        s2 = self.a2
        for ind, i in enumerate(s1):

            lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
            upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

            if i > upper_bound:
                LB_sum = LB_sum + (i - upper_bound) ** 2
            elif i < lower_bound:
                LB_sum = LB_sum + (i - lower_bound) ** 2

        return np.sqrt(LB_sum)

    @staticmethod
    def dist(n, m):
        return abs(n-m)

    def solve(self, i : int, j : int):
        #if (DTW.dist(self.a1[i],self.a2[j]) > self.bound):
        if (abs(i-j) > self.bound):
            return self.infty

        if self.m[i,j] >= 0:
            return self.m[i,j]
        if i == 0 and j == 0:
            self.m[i,j] = DTW.dist(self.a1[i],self.a2[j])
            return self.m[i,j]

        if i == 0:
            self.m[i,j] = DTW.dist(self.a1[i], self.a2[j]) + self.solve(i, j-1)
            return self.m[i,j]
        if j == 0:
            self.m[i,j] = DTW.dist(self.a1[i], self.a2[j]) + self.solve(i-1, j)
            return self.m[i,j]

        self.m[i, j] = DTW.dist(self.a1[i], self.a2[j]) + min(self.solve(i-1, j),self.solve(i, j-1),self.solve(i-1, j-1))
        return self.m[i,j]


class KNN:

    def __init__(self, k):
        self.k = k

    def train(self, tsss, labels):
        # tsss is list of arrays where the array is a 9 by width combination of all signal data for a given frame
        assert(len(tsss) == len(labels))
        self.tsss = np.array(tsss)
        self.labels = np.array(labels)


    def comparesingle(self, s1, s2):
        return DTW(s1,s2).result

    def compareall(self, tss1, tss2):
        assert(len(tss1) == len(tss2))
        return np.sqrt(sum([self.comparesingle(tss1[i], tss2[i])**2 for i in range(len(tss1))]))

    def predict(self, test):
        comparefunc = functools.partial(self.compareall, tss2=test)
        p = Pool(processes=8)
        return Counter(np.array(sorted(zip(p.map(comparefunc, self.tsss), self.labels)))[:self.k,1].astype('int')).most_common(1)[0][0]


    def predictmany(self, testset):
        pass




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



if __name__ == '__main__':
    model = KNN(10)
    model.train(trainset, trainlabels)
    res = model.predict(testset[0])
    print(res)
    print(testlabels[0])

#a1 = np.array([1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])
#a2 = np.array([2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,98,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9])

# a1 = np.array([i for i in range(0,1000,1)])
# a2 = np.array([i for i in range(1000,10,-1)])
#
# print(len(a1))
# print(len(a2))
# print(DTW(a1,a2).result)
