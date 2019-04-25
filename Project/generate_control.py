import pandas as pd
import numpy as np
import time
import datetime

class LabelingException(Exception):
    pass


def addthirtyseconds(s : str):
    t = datetime.datetime.strptime(s, "%H:%M:%S")
    t += datetime.timedelta(seconds=30)
    return datetime.datetime.strftime(t, "%H:%M:%S")

# print(thirtyseconds("23:59:48"))


def process(k):
    df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data{}.txt".format(k), sep=' ', header=None, names=range(11), low_memory=False, quotechar="'")
    data = df.iloc[:, [0,3,4]].values
    print(data[78828])

    labels = np.zeros((data.shape[0],1))
    first_occurrence = True
    isAF = False
    for i in range(data.shape[0]):

        row = data[i]

        if row[1] == "Pause" or row[1] == "MB":
            labels[i] = -1
        elif isAF:
            labels[i] = 1
        else:
            labels[i] = 0

        if (isinstance(row[2], float)):
            continue

        if "START AF" in row[2]:
            if isAF:
                raise LabelingException("Found orphaned START AF")
            isAF = True
            if i > 0 and labels[i] != -1:
                labels[i] = 1
            first_occurrence = False

        if "END AF" in row[2]:
            if not isAF:
                if first_occurrence:
                    for j in range(i+1):
                        if labels[i] == 1:
                            raise LabelingException("Cannot be first occurrence")
                        if labels[i] == 0:
                            labels[i] = 1
                else:
                    raise LabelingException("Found orphaned END AF")
            isAF = False
            first_occurrence = False

    blocks = []
    timestart = data[0][0]
    timeend = addthirtyseconds(timestart)
    containsnegative = False
    af_sum = 0
    total = 0

    for i in range(data.shape[0]):
        row = data[i]

        if (timestart < timeend and row[0] >= timeend) or (
                timestart > timeend and row[0] > timeend and row[0] < timestart):
            if containsnegative:
                blocks.append((timestart, -1))
            elif (af_sum / total >= 0.75):
                blocks.append((timestart, 1))
            else:
                blocks.append((timestart, 0))

            timestart = timeend
            timeend = addthirtyseconds(timestart)
            containsnegative = False
            af_sum = 0
            total = 0

        if labels[i] == -1:
            containsnegative = True
        total += 1
        af_sum += labels[i]

    with open("AF-Raw-Data/AF_Data/Class2/Control{}.txt".format(k), 'w') as f:
        for (t,v) in blocks:
            f.write("{} {}\n".format(t, v))


process(2)