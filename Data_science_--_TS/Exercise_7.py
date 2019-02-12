import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sgn
from Exercise_6 import fourier


def highpass(series: pd.Series)->pd.Series:
    b, a = sgn.butter()
    v,w = sgn.freqs(b, a)
    return


def lowpass(series: pd.Series)->pd.Series:
    return


def bandpass(series: pd.Series)->pd.Series:
    return
