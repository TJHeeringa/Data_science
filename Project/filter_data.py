import pandas as pd

sample_threshold = 10

for i in range(65, 66):
    df = pd.read_csv("AF_Feature_Data/Data{}.csv".format(i), sep=' ', index_col=False, usecols=range(36),
                     low_memory=False)
    classes = pd.read_csv("AF-Raw-Data/AF_Data/Class/Control{}.txt".format(i), sep=' ', header=None,
                          index_col=False, names=["time", "NaN", "label"], low_memory=False, usecols=["label"],
                          dtype={"label": 'Int64'})
    print(df.columns)
    print(classes.columns)
    data = pd.concat([df, classes], axis=1)
    print(data.columns)
    # data = data[data.values[:, 0] >= sample_threshold]
    data.to_csv("AF_Filtered_Data/Data{}.csv".format(i), sep=' ', header=True)
