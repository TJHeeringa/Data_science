import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sgn
from Exercise_6 import fourier



def highpass(series: pd.Series, freq)->pd.Series:
    b, a = sgn.butter(2, freq, btype="high")
    v, w = sgn.freqs(b, a)
    return


def lowpass(series: pd.Series, freq)->pd.Series:
    b, a = sgn.butter(2, freq, btype="low")
    v, w = sgn.freqs(b, a)
    return


def bandpass(series: pd.Series, freqs)->pd.Series:
    b, a = sgn.butter(2, freqs, btype="band")
    v, w = sgn.freqs(b, a)
    return
