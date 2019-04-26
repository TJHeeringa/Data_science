import pandas as pd
import numpy as np
from scipy import stats as st
import csv
import time
from multiprocessing import Pool


def collect_values(series, t2, t1, index):
    values_between_time_stamps = []

    while index < series.shape[0] and '' != series[index,0] and (series[index,0] < t2 or t2 < t1 <= series[index,0]):
        values_between_time_stamps.append(series[index,1])
        index += 1
    return np.array(values_between_time_stamps), index

#
# def collect_values_old(series, t2, t1, index):
#     values_between_time_stamps = []
#     count = 0
#     for k, (_time, value) in enumerate(zip(
#             series.values[index:, 0], series.values[index:, 1],
#     )):
#         count = k
#         if _time == '':
#
#             break
#         elif _time < t2 or t2 < t1 <= _time:
#             values_between_time_stamps.append(value)
#         else:
#
#             break
#     index += count
#     return np.array(values_between_time_stamps), index


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def process(i):
    start_time = time.time()

    # Load data from file
    df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data{}.txt".format(i), sep=' ', header=None, names=range(11), low_memory=False)
    classes = pd.read_csv("AF-Raw-Data/AF_Data/Class2/Control{}.txt".format(i), sep=' ', header=None, names=range(11), low_memory=False)

    # print("{:.3}: loading data done.".format(time.time() - start_time))

    # parameters
    bound = 2000
    window = 10

    # Pick relevant columns
    time_stamps = classes.iloc[:, 0]
    R_peak = df.iloc[:, [0, 1]]

    # print("{:.3}: column filter done.".format(time.time() - start_time))

    # shift time stamps
    # time_stamps_shifted = time_stamps.shift(-1, fill_value=time_stamps.values[0])

    # print("{:.3}: shifting data done.".format(time.time() - start_time))

    # Prefilter R_peak
    R_peak_bounded = R_peak[R_peak.values[:, 1] < bound]

    R_peak_bounded_rolling_mean = R_peak_bounded.copy()
    R_peak_bounded_rolling_mean.iloc[:, 1] = R_peak_bounded_rolling_mean.iloc[:, 1].rolling(window=window).mean()
    R_peak_bounded_rolling_mean = R_peak_bounded_rolling_mean[window:-window]

    R_peak_bounded_differencing = R_peak_bounded.copy()
    R_peak_bounded_differencing.iloc[:, 1] = R_peak_bounded_differencing.iloc[:, 1] - R_peak_bounded_differencing.iloc[:, 1].shift()
    R_peak_bounded_differencing = R_peak_bounded_differencing[1:]

    # print("{:.3}: prefiltering data done.".format(time.time() - start_time))

    with open('AF_Feature_Data_test/Data{}.csv'.format(i), 'w+', newline='') as csvfile:
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
        # index_o_old = 0
        # index_rm_old = 0
        # index_d_old = 0
        # print("{:.3}: writing header done.".format(time.time() - start_time))

        collecting_time = 0.
        quantile_time = 0.
        writing_time = 0.

        R_peak_bounded_vals = R_peak_bounded.values
        R_peak_bounded_rolling_mean_vals = R_peak_bounded_rolling_mean.values
        R_peak_bounded_differencing_vals = R_peak_bounded_differencing.values
        timestamp_vals = time_stamps.values
        timestamp_length = len(timestamp_vals)

        for j in range(timestamp_length):
            if j == timestamp_length-1:
                t1 = timestamp_vals[j]
                t2 = timestamp_vals[0]
            else:
                t1 = timestamp_vals[j]
                t2 = timestamp_vals[j+1]

            collect_start = time.time()
            (values_between_time_stamps_o, index_o) = collect_values(R_peak_bounded_vals, t2, t1, index_o)
            (values_between_time_stamps_rm, index_rm) = collect_values(R_peak_bounded_rolling_mean_vals, t2, t1, index_rm)
            (values_between_time_stamps_d, index_d) = collect_values(R_peak_bounded_differencing_vals, t2, t1, index_d)

            # (values_between_time_stamps_o_old, index_o_old) = collect_values_old(R_peak_bounded, t2, t1, index_o_old)
            # (values_between_time_stamps_rm_old, index_rm_old) = collect_values_old(R_peak_bounded_rolling_mean, t2, t1,index_rm_old)
            # (values_between_time_stamps_d_old, index_d_old) = collect_values_old(R_peak_bounded_differencing, t2, t1, index_d_old)

# #            assert (values_between_time_stamps_o == values_between_time_stamps_o_old)
#             assert (index_o == index_o_old)
# #            assert (values_between_time_stamps_rm == values_between_time_stamps_rm_old)
#             assert (index_rm == index_rm_old)
# #            assert (values_between_time_stamps_d == values_between_time_stamps_d_old)
#             assert (index_d == index_d_old)

            if values_between_time_stamps_o.size == 0:
                values_between_time_stamps_o = [0]
            if values_between_time_stamps_rm.size == 0:
                values_between_time_stamps_rm = [0]
            if values_between_time_stamps_d.size == 0:
                values_between_time_stamps_d = [0]

            collecting_time += time.time() - collect_start

            quant_start = time.time()
            quantiles_o = np.percentile(values_between_time_stamps_o, [25, 50, 75])
            quantiles_d = np.percentile(values_between_time_stamps_d, [25, 50, 75])
            quantiles_rm = np.percentile(values_between_time_stamps_rm, [25, 50, 75])

            quantile_time += time.time() - quant_start

            write_start = time.time()
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
            writing_time += time.time()-write_start
    end_time = time.time()
    print("iteration {} took {:.06} seconds. collecting: {:.06}, quanting: {:.06}, writing: {:.06}".format(i, end_time-start_time, collecting_time, quantile_time, writing_time))
    # print("collecting: {:.3}, quanting: {:.3}, writing: {:.3}". format(collecting_time, quantile_time, writing_time))


threads = 4
start = 1
end = 805

if __name__ == "__main__":
    p = Pool(threads)
    l = [i for i in range(start, end)]
    p.map(process, l)
    # for i in range(start, end):
    #     process(i)


