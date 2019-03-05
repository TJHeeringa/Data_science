from Exercise_6 import fourier
from Exercise_5 import activities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sgn


def highpass(series: pd.Series, freq, fs, order=2)->pd.Series:
    nyq = 0.5 * fs
    freq /= nyq
    b, a = sgn.butter(order, freq, btype='high')
    return pd.Series(sgn.lfilter(b, a, series.values))


def lowpass(series: pd.Series, freq, fs, order=2)->pd.Series:
    nyq = 0.5 * fs
    freq /= nyq
    b, a = sgn.butter(order, freq, btype='low')
    return pd.Series(sgn.lfilter(b, a, series.values))


def bandpass(series: pd.Series, low, high, fs, order=2)->pd.Series:
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = sgn.butter(order, [low, high], btype='band')
    return pd.Series(sgn.lfilter(b, a, series.values))


plt.figure()
for activity in activities:
    plt.plot(fourier(activity["val"]))
plt.title("Fourier plot of activities")
plt.legend(["activity {}".format(activity['label'].values[0]) for activity in activities])
plt.show()
