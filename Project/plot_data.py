import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("AF-Raw-Data/AF_Data/ECG_data/Data2.txt", sep=' ', header=None, names=range(11), low_memory=False)
print(df.info())
print(df.head())
print(df.columns)

R_peak = df.iloc[:, 1]

differencing = R_peak - R_peak.shift()
differencing.dropna(inplace=True)

R_peak_bounded = R_peak[R_peak.values < 2000]
differencing_bounded = differencing[np.abs(differencing.values) < 1000]

plt.figure()
plt.plot(differencing_bounded.values)

plt.figure()
plt.plot(differencing_bounded.values[:2200])

plt.figure()
plt.plot(R_peak_bounded.values)

plt.figure()
plt.plot(R_peak_bounded.values[:2200])

plt.show()
