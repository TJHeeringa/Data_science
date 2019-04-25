import pandas as pd
import numpy as np

def process(i):
    df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data{}.txt".format(i), sep=' ', header=None, names=range(11), low_memory=False, quotechar="'")
    data = df.iloc[:, [0,1,3,4]].values
    labels = np.zeros((data.shape[0],1))
    isAF = False
    for i in range(data.shape[0]):
        row = data[i]
        if row[3]
    print(data[78828])

process(2)