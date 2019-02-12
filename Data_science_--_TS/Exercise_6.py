from scipy.fftpack import fft, fftshift
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sgn


def fourier(series: pd.Series, shifted_to_center=True)->pd.Series:
    if shifted_to_center:
        return pd.Series(fftshift(fft(series.values)))
    else:
        return pd.Series(fft(series.values))


x = np.arange(5000)
test_signal = np.sin(2*2*np.pi*x)+0.8*np.sin(4*2*np.pi*x)+0.5*np.sin(6*2*np.pi*x)
test_signal2 = 2+np.sin(2*2*np.pi*x)+0.8*np.sin(4*2*np.pi*x)+0.5*np.sin(6*2*np.pi*x)

a = fft(test_signal)
b = fourier(pd.Series(test_signal), shifted_to_center=False)
a_shifted = fftshift(fft(test_signal))
b_shifted = fourier(pd.Series(test_signal))

a2 = fft(test_signal2)
b2 = fourier(pd.Series(test_signal2), shifted_to_center=False)
a2_shifted = fftshift(fft(test_signal2))
b2_shifted = fourier(pd.Series(test_signal2))

test_signal2_offsetless = test_signal2-np.mean(test_signal2)
a3 = fft(test_signal2_offsetless)
b3 = fourier(pd.Series(test_signal2_offsetless), shifted_to_center=False)
a3_shifted = fftshift(fft(test_signal2_offsetless))
b3_shifted = fourier(pd.Series(test_signal2_offsetless))

plt.figure()
plt.title("Unshifted")
plt.plot(a)
plt.plot(b)
plt.legend(["fft", "fourier"])

plt.figure()
plt.title("Shifted")
plt.plot(a_shifted)
plt.plot(b_shifted)
plt.legend(["fft", "fourier"])

plt.figure()
plt.title("Unshifted with offset")
plt.plot(a2)
plt.plot(b2)
plt.legend(["fft", "fourier"])

plt.figure()
plt.title("Shifted with offset")
plt.plot(a2_shifted)
plt.plot(b2_shifted)
plt.legend(["fft", "fourier"])

plt.figure()
plt.title("Unshifted with offset")
plt.plot(a3)
plt.plot(b3)
plt.legend(["fft", "fourier"])

plt.figure()
plt.title("Shifted with offset")
plt.plot(a3_shifted)
plt.plot(b3_shifted)
plt.legend(["fft", "fourier"])

plt.show()

activities = []
for i, activity in enumerate(activities):
    plt.figure()
    plt.title("Activity {}".format(i))
    plt.plot(fourier(activity))
plt.show()
