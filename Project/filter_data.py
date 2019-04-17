import pandas as pd

sample_threshold = 10

for i in range(1, 805):
    df = pd.read_csv("AF_Feature_Data/Data{}.csv".format(i), sep=' ', index_col=False, usecols=range(36),
                     low_memory=False)
    classes = pd.read_csv("AF-Raw-Data/AF_Data/Class/Control{}.txt".format(i), sep='\s+', header=None,
                          index_col=False, names=["time", "label"], low_memory=False, usecols=["label"],
                          dtype={"label": 'Int64'})
    data = pd.concat([df, classes], axis=1)
    data = data[data.values[:, 0] >= sample_threshold]
    data.to_csv("AF_Filtered_Data/Data{}.csv".format(i), sep=' ', header=True)
