import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close("all")

df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data65.txt", sep=' ', header=None, names=range(11), low_memory=False)
# print(df.info())
# print(df.head())
# print(df.columns)
#
# print(df.iloc[83489:83490, :])
# print(df.iloc[33950:34000, :])
R_peak = df.iloc[:, [1, 5]]


R_peak_bounded = R_peak[R_peak.values[:, 0] < 1000]
AF_indeces = R_peak_bounded[["AF" in str(x) for x in R_peak_bounded.iloc[:, 1]]].index
print(AF_indeces)

window = 10

R_peak_rolling_mean = R_peak.copy()
R_peak_rolling_mean.iloc[:, 0] = R_peak_rolling_mean.iloc[:, 0].rolling(window=window).mean()
R_peak_rolling_mean = R_peak_rolling_mean.iloc[window:, :]
R_peak_rolling_mean = R_peak_rolling_mean[R_peak_rolling_mean.values[:, 0] < 1200]

AF_indeces = R_peak_rolling_mean[["AF" in str(x) for x in R_peak_rolling_mean.iloc[:, 1]]].index

R_peak_bounded = R_peak[np.abs(R_peak.values[:, 0]) < 1200]
R_peak_bounded_rolling_mean = R_peak_bounded.copy()
R_peak_bounded_rolling_mean.iloc[:, 0] = R_peak_bounded_rolling_mean.iloc[:, 0].rolling(window=window).mean()


plt.figure()
plt.plot(R_peak.index, R_peak.values[:, 0])
plt.title("Original")
for index in AF_indeces:
    plt.axvline(x=index, color="red")

plt.figure()
plt.plot(R_peak_rolling_mean.index, R_peak_rolling_mean.values[:, 0])
plt.title("Rolling average {}".format(window))
for index in AF_indeces:
    plt.axvline(x=index, color="red")

plt.figure()
plt.plot(R_peak_bounded.index, R_peak_bounded.values[:, 0])
plt.title("Bounded")
for index in AF_indeces:
    plt.axvline(x=index, color="red")

plt.figure()
plt.plot(R_peak_bounded_rolling_mean.index, R_peak_bounded_rolling_mean.values[:, 0])
plt.title("Bounded rolling average {}".format(window))
for index in AF_indeces:
    plt.axvline(x=index, color="red")

plt.figure()
plt.plot(R_peak_bounded_rolling_mean.index, R_peak_bounded_rolling_mean.values[:, 0])
plt.title("Zoomed bounded rolling average {}".format(window))
for index in AF_indeces:
    plt.axvline(x=index, color="red")
plt.xlim([min(AF_indeces)-1000, max(AF_indeces)+1000])

plt.show()
