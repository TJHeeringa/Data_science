from Exercise_6 import fourier, plotfourier
from Exercise_5 import activities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sgn


class Cseries:
    @classmethod
    def highpass(cls, series: pd.Series, freq, fs, order=2)->pd.Series:
        nyq = 0.5 * fs
        freq /= nyq
        b, a = sgn.butter(order, freq, btype='high')
        series.loc[:, "val"] = sgn.lfilter(b, a, series["val"])
        return series
        # return pd.Series(sgn.lfilter(b, a, series.values))

    @classmethod
    def lowpass(cls, series: pd.Series, freq, fs, order=2)->pd.Series:
        nyq = 0.5 * fs
        freq /= nyq
        b, a = sgn.butter(order, freq, btype='low')
        return pd.Series(sgn.lfilter(b, a, series.values))

    @classmethod
    def bandpass(cls, series: pd.Series, low, high, fs, order=2)->pd.Series:
        nyq = 0.5 * fs
        low /= nyq
        high /= nyq
        b, a = sgn.butter(order, [low, high], btype='band')
        series.loc[:, "val"] = sgn.lfilter(b, a, series["val"])
        return series
        # return pd.Series(sgn.lfilter(b, a, series.values))


def highpass(values, freq, fs, order=2):
    nyq = 0.5 * fs
    freq /= nyq
    b, a = sgn.butter(order, freq, btype='high')
    return sgn.lfilter(b, a, values)


def lowpass(values, freq, fs, order=2):
    nyq = 0.5 * fs
    freq /= nyq
    b, a = sgn.butter(order, freq, btype='low')
    return sgn.lfilter(b, a, values)


def bandpass(values, low, high, fs, order=2):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = sgn.butter(order, [low, high], btype='band')
    return sgn.lfilter(b, a, values)


plt.figure()
for activity in activities:
    plotfourier(len(activity["val"]), 1.0 / 50.0, activity["val"], label=activity["label"].values[0])
plt.title("Fourier plot of activities")
plt.legend()
plt.show()

plt.figure()
for activity in activities:
    filtered_activity = bandpass(activity["val"], 2, 3, 50)
    plotfourier(len(activity["val"]), 1.0 / 50.0, filtered_activity, label=activity["label"].values[0])
plt.title("bandpass2")
plt.legend()

# plt.figure()
# for activity in activities:
#     activity = Cseries.bandpass(activity, 2, 3, 50)
#     plotfourier(len(activity["val"]), 1.0 / 50.0, activity["val"], label=activity["label"].values[0])
# plt.title("bandpass3")
# plt.legend()

plt.show()
