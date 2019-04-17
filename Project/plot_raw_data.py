import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.close("all")

# df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data23.txt", sep=' ', header=None, names=range(11), low_memory=False)
df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data65.txt", sep=' ', header=None, names=range(11), low_memory=False)
# print(df.info())
# print(df.head())
# print(df.columns)
#
# print(df.iloc[83489:83490, :])
# print(df.iloc[33950:34000, :])
R_peak = df.iloc[:, [1, 5]]

bound = 2000

R_peak_bounded = R_peak[R_peak.values[:, 0] < bound]
AF_indeces = R_peak_bounded[["AF" in str(x) for x in R_peak_bounded.iloc[:, 1]]].index
print(AF_indeces)

window = 10
# min_plot = min(AF_indeces)-1000
# max_plot = max(AF_indeces)+1000
min_plot = 30000
max_plot = 31000
# min_plot = 33800
# max_plot = 34500
# min_plot = 24000
# max_plot = 26000

R_peak_rolling_mean = R_peak.copy()
R_peak_rolling_mean.iloc[:, 0] = R_peak_rolling_mean.iloc[:, 0].rolling(window=window).mean()
R_peak_rolling_mean = R_peak_rolling_mean.iloc[window:, :]
R_peak_rolling_mean = R_peak_rolling_mean[R_peak_rolling_mean.values[:, 0] < bound]

AF_indeces = R_peak_rolling_mean[["AF" in str(x) for x in R_peak_rolling_mean.iloc[:, 1]]].index

R_peak_bounded_rolling_mean = R_peak_bounded.copy()
R_peak_bounded_rolling_mean.iloc[:, 0] = R_peak_bounded_rolling_mean.iloc[:, 0].rolling(window=window).mean()

R_peak_bounded_differencing = R_peak_bounded.copy()
R_peak_bounded_differencing.iloc[:, 0] = R_peak_bounded_differencing.iloc[:, 0]-R_peak_bounded_differencing.iloc[:, 0].shift()
R_peak_bounded_differencing = R_peak_bounded_differencing[1:]

plt.figure()
plt.plot(R_peak.index, R_peak.values[:, 0])
plt.title("Original")
for index in AF_indeces:
    plt.axvline(x=index, color="red")
plt.ylim([0, 5000])
plt.xlim([min_plot, max_plot])

plt.figure()
plt.plot(R_peak_rolling_mean.index, R_peak_rolling_mean.values[:, 0])
plt.title("Rolling average {}".format(window))
for index in AF_indeces:
    plt.axvline(x=index+window, color="red")
plt.xlim([min_plot, max_plot])

plt.figure()
plt.plot(R_peak_bounded.index, R_peak_bounded.values[:, 0])
plt.title("Bounded")
for index in AF_indeces:
    plt.axvline(x=index, color="red")
plt.xlim([min_plot, max_plot])

plt.figure()
plt.plot(R_peak_bounded_rolling_mean.index, R_peak_bounded_rolling_mean.values[:, 0])
plt.title("Bounded rolling average {}".format(window))
for index in AF_indeces:
    plt.axvline(x=index+window, color="red")
plt.xlim([min_plot, max_plot])

plt.figure()
plt.plot(R_peak_bounded_differencing.index, R_peak_bounded_differencing.values[:, 0])
plt.title("Bounded first order difference")
for index in AF_indeces:
    plt.axvline(x=index+1, color="red")
plt.xlim([min_plot, max_plot])

plt.show()
