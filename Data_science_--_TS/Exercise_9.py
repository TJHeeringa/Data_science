import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("climate-data/GlobalLandTemperaturesByCountry.csv", sep=',')

data = df.values

countries = {}

for entry in data:
    if entry[3] not in countries:
        countries[entry[3]] = []
    countries[entry[3]].append(entry[1])

ls = ["Norway", "Singapore", "Finland", "Cambodia"]

plt.figure()
for name in ls:
    temps = countries[name]
    plt.plot(range(-len(temps),0), temps)
plt.legend(ls)

plt.show()
