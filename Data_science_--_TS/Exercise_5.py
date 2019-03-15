from Exercise_3 import ACTIVITY_DICT
from Exercise_4 import merged
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

activities = [merged[merged['label'] == i] for i in range(1, 7)]

characteristics = []

for activity in activities:
    characteristic = {
        "label": ACTIVITY_DICT[activity["label"].values[0]],
        "mean": activity["val"].mean(),
        "variance": activity["val"].var(),
        "skewness": activity["val"].skew(),
        "kurtosis": activity["val"].kurtosis(),
        "spread": activity["val"].max()-activity["val"].min(),
    }
    characteristics = np.append(characteristics, characteristic)

if __name__ == '__main__':
    plt.figure()
    x = np.arange(-1, 1, 0.001)
    for characteristic in characteristics:
        plt.plot(x, mlab.normpdf(x, characteristic['mean'], math.sqrt(characteristic['variance'])))
    plt.title("Gaussian plot of activities")
    plt.legend(["{}".format(characteristic['label']) for characteristic in characteristics])
    plt.show()

    print("Based on the plot for the charactestics we selected, it is clear that one can discern")
    print("between active and passive activities.")


