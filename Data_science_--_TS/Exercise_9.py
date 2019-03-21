import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.seasonal import seasonal_decompose


def dickey_fuller(_dict):
    for key, value in _dict.items():
        result = ts.adfuller(value)
        print('Augmented Dicky-Fuller for: {}'.format(key))
        print(' -------------------------------------- ')
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for result_key, result_value in result[4].items():
            print('\t%s: %.3f' % (result_key, result_value))
        print('')


df = pd.read_csv("climate-data/GlobalLandTemperaturesByCountry.csv", sep=',')

countries = {}

for entry in df.values:
    if entry[3] not in countries:
        countries[entry[3]] = []
    countries[entry[3]].append(entry[1])

ls = ["Cambodia", "Finland", "Norway", "Singapore"]
countries = {k: v for (k, v) in countries.items() if k in ls}

plt.figure()
for country, temperature in countries.items():
    plt.plot(range(-len(temperature), 0), temperature)
plt.legend(ls)
plt.show()


shortest_hist = min(map(len, countries.values()))
nan_index = shortest_hist
for country, temps in countries.items():
    # range from two, since temps[-1] = NaN for all country in countries
    for k in range(2, shortest_hist):
        if np.isnan(temps[-k]):
            if k < nan_index:
                nan_index = k
            break
countries = {k: v[-nan_index+1:-2] for (k, v) in countries.items()}

# TODO use DTW on this
dickey_fuller(countries)

moving_avg_countries = dict()
differencing_countries = dict()
decomposing_countries = dict()
for (country, temperatures) in countries.items():
    temps_series = pd.Series(temperatures)
    # moving_avg_countries
    moving_avg = pd.rolling_mean(pd.Series(temps_series), 12)
    moving_avg.dropna(inplace=True)
    moving_avg_countries[country] = moving_avg
    # differencing_countries
    differencing = temps_series - temps_series.shift()
    differencing.dropna(inplace=True)
    differencing_countries[country] = differencing
    # decomposing countries
    decomposition = seasonal_decompose(temps_series)
    decomposing = decomposition.resid
    decomposing.dropna(inplace=True)
    decomposing_countries[country] = decomposing

# TODO use DTW on these
dickey_fuller(moving_avg_countries)
dickey_fuller(differencing_countries)
dickey_fuller(decomposing_countries)



