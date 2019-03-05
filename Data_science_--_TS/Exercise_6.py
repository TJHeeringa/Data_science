from scipy.fftpack import fft, fftshift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Exercise_5 import activities


def fourier(series: pd.Series, shifted_to_center=True)->pd.Series:
    if shifted_to_center:
        return pd.Series(fftshift(fft(series.values)))
    else:
        return pd.Series(fft(series.values))

def plotfourier(num_samples, sample_spacing, values):
    yf = fft(values)
    xf = np.linspace(0.0, 1.0 / (2.0 * sample_spacing), num_samples // 2)
    plt.plot(xf, 2.0 / num_samples * np.abs(yf[0:num_samples // 2]))



if __name__ == "__main__":

    num_samples = 5000
    sample_spacing = 1.0/1000.0

    x = np.linspace(0.0, num_samples*sample_spacing, num_samples)
    testsignal = np.sin(2*2*np.pi*x)+0.8*np.sin(4*2*np.pi*x)+0.5*np.sin(6*2*np.pi*x)

    for signal in activities:
        normed = signal["val"] - signal["val"].mean()

        # plt.figure()
        # plt.title(signal["label"].values[0])
        # plt.plot(signal["val"])
        # plt.legend([signal["label"].values[0]])

        plt.figure()
        plt.title("FFT of " + str(signal["label"].values[0]))
        plt.ylim(0.,0.04)
        plotfourier(len(normed), 1.0/50.0, normed)
        plt.legend(["fft"])

    plt.show()