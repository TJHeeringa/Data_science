import pandas as pd
import numpy as np
from scipy import stats as st
import csv
import time
from multiprocessing import Pool



def collect_values(series, t2, t1, index):
    values_between_time_stamps = np.empty(0)
    for k, (_time, value) in enumerate(zip(
            series.values[index:, 0], series.values[index:, 1],
    )):
        if _time == '':
            break
        elif t2 < t1 <= _time:
            values_between_time_stamps = np.append(values_between_time_stamps, value)
        elif _time < t2:
            values_between_time_stamps = np.append(values_between_time_stamps, value)
        else:
            index += k
            break
    return values_between_time_stamps, index


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def process(i):
    # Load data from file
    df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data{}.txt".format(i), sep=' ', header=None, names=range(11), low_memory=False)
    classes = pd.read_csv("AF-Raw-Data/AF_Data/Class/Control{}.txt".format(i), sep=' ', header=None, names=range(11), low_memory=False)

    # parameters
    bound = 2000
    window = 10

    # Pick relevant columns
    time_stamps = classes.iloc[:, 0]
    R_peak = df.iloc[:, [0, 1, 5]]

    # shift time stamps
    time_stamps_shifted = time_stamps.shift(-1, fill_value=time_stamps.values[0])

    # Prefilter R_peak
    R_peak_bounded = R_peak[R_peak.values[:, 1] < bound]

    R_peak_bounded_rolling_mean = R_peak_bounded.copy()
    R_peak_bounded_rolling_mean.iloc[:, 1] = R_peak_bounded_rolling_mean.iloc[:, 1].rolling(window=window).mean()
    R_peak_bounded_rolling_mean = R_peak_bounded_rolling_mean[window:-window]

    R_peak_bounded_differencing = R_peak_bounded.copy()
    R_peak_bounded_differencing.iloc[:, 1] = R_peak_bounded_differencing.iloc[:, 1] - R_peak_bounded_differencing.iloc[:, 1].shift()
    R_peak_bounded_differencing = R_peak_bounded_differencing[1:]

    start_time = time.time()
    with open('AF_Feature_Data/Data{}.csv'.format(i), 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([
            "samples",
            "o - mean",
            "o - var",
            "o - max",
            "o - min",
            "o - spread",
            "o - mad",
            "o - kurtosis",
            "o - skew",
            "o - median",
            "o - first quantile",
            "o - last quantile",
            "o - sum",
            "d - mean",
            "d - var",
            "d - max",
            "d - min",
            "d - spread",
            "d - mad",
            "d - kurtosis",
            "d - skew",
            "d - median",
            "d - first quantile",
            "d - last quantile",
            "d - sum",
            "rm - mean",
            "rm - var",
            "rm - max",
            "rm - min",
            "rm - spread",
            "rm - mad",
            "rm - kurtosis",
            "rm - skew",
            "rm - median",
            "rm - first quantile",
            "rm - last quantile",
            "rm - sum"
        ])
        index_o = 0
        index_rm = 0
        index_d = 0
        for (t1, t2) in zip(time_stamps.values, time_stamps_shifted.values):
            [values_between_time_stamps_o, index_o] = collect_values(R_peak_bounded, t2, t1, index_o)
            [values_between_time_stamps_rm, index_rm] = collect_values(R_peak_bounded_rolling_mean, t2, t1, index_rm)
            [values_between_time_stamps_d, index_d] = collect_values(R_peak_bounded_differencing, t2, t1, index_d)
            if values_between_time_stamps_o.size == 0:
                values_between_time_stamps_o = [0]
            if values_between_time_stamps_rm.size == 0:
                values_between_time_stamps_rm = [0]
            if values_between_time_stamps_d.size == 0:
                values_between_time_stamps_d = [0]
            quantiles_o = np.percentile(values_between_time_stamps_o, [25, 50, 75])
            quantiles_d = np.percentile(values_between_time_stamps_d, [25, 50, 75])
            quantiles_rm = np.percentile(values_between_time_stamps_rm, [25, 50, 75])

            spamwriter.writerow([
                np.min([len(values_between_time_stamps_o), len(values_between_time_stamps_rm), len(values_between_time_stamps_d)]),
                np.mean(values_between_time_stamps_o),
                np.var(values_between_time_stamps_o),
                np.max(values_between_time_stamps_o),
                np.min(values_between_time_stamps_o),
                np.max(values_between_time_stamps_o)-np.min(values_between_time_stamps_o),
                mad(values_between_time_stamps_o),
                st.kurtosis(values_between_time_stamps_o),
                st.skew(values_between_time_stamps_o),
                quantiles_o[1],
                quantiles_o[0],
                quantiles_o[2],
                np.sum(values_between_time_stamps_o),
                np.mean(values_between_time_stamps_d),
                np.var(values_between_time_stamps_d),
                np.max(values_between_time_stamps_d),
                np.min(values_between_time_stamps_d),
                np.max(values_between_time_stamps_d) - np.min(values_between_time_stamps_d),
                mad(values_between_time_stamps_d),
                st.kurtosis(values_between_time_stamps_d),
                st.skew(values_between_time_stamps_d),
                quantiles_d[1],
                quantiles_d[0],
                quantiles_d[2],
                np.sum(values_between_time_stamps_d),
                np.mean(values_between_time_stamps_rm),
                np.var(values_between_time_stamps_rm),
                np.max(values_between_time_stamps_rm),
                np.min(values_between_time_stamps_rm),
                np.max(values_between_time_stamps_rm) - np.min(values_between_time_stamps_rm),
                mad(values_between_time_stamps_rm),
                st.kurtosis(values_between_time_stamps_rm),
                st.skew(values_between_time_stamps_rm),
                quantiles_rm[1],
                quantiles_rm[0],
                quantiles_rm[2],
                np.sum(values_between_time_stamps_rm),
            ])
    end_time = time.time()
    print("iteration {} took {} seconds".format(i, end_time-start_time))


threads = 12
start = 0
end = 804

if __name__ == "__main__":
    p = Pool(threads)
    l = [i for i in range(start, end)]
    p.map(process, l)

