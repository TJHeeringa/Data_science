import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA


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
    # moving_avg = pd.rolling_mean(pd.Series(temps_series), 12)
    # moving_avg = pd.Series(temps_series).rolling(12).mean()
    # moving_avg.dropna(inplace=True)
    # moving_avg_countries[country] = moving_avg
    # # differencing_countries
    differencing = temps_series - temps_series.shift()
    differencing.dropna(inplace=True)
    differencing_countries[country] = differencing
    # # decomposing countries
    # decomposition = seasonal_decompose(temps_series)
    # decomposing = decomposition.resid
    # decomposing.dropna(inplace=True)
    # decomposing_countries[country] = decomposing

# TODO use DTW on these
dickey_fuller(moving_avg_countries)
dickey_fuller(differencing_countries)
dickey_fuller(decomposing_countries)

plt.figure()
for (country, temperature) in differencing_countries.items():
    acf = ts.acf(temperature, nlags=20)
    plt.title('Autocorrelation Function')
    plt.plot(acf, label="{}".format(country))
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(temperature)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(temperature)), linestyle='--', color='gray')
    plt.xlabel("Lag")
plt.legend()

plt.figure()
for (country, temperature) in differencing_countries.items():
    pacf = ts.pacf(temperature, nlags=20, method='ols')
    plt.title('Partial Autocorrelation Function')
    plt.plot(pacf, label="{}".format(country))
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96 / np.sqrt(len(temperature)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(temperature)), linestyle='--', color='gray')
    plt.xlabel("Lag")
plt.legend()

norway_temperature = differencing_countries["Norway"]
model = ARIMA(norway_temperature, order=(2, 0, 0))
results_AR = model.fit(disp=-1)
model = ARIMA(norway_temperature, order=(0, 0, 2))
results_MA = model.fit(disp=-1)
for x in range(0, 4):
    for y in range(0, 4):
        for z in range(0, 4):
            try:
                model = ARIMA(norway_temperature, order=(x, y, z))
                results_ARIMA = model.fit(disp=-1)
                print("{},{},{}".format(x, y, z))
            except ValueError:
                pass

number_of_months_to_predict = 7
x = np.linspace(2012-len(norway_temperature)/12, 2012, len(norway_temperature))
p = np.linspace(2012, 2012+number_of_months_to_predict/12, number_of_months_to_predict+1)

plt.figure()
plt.plot(x, norway_temperature, label="Original")
plt.plot(x, results_AR.fittedvalues, label='AR')
plt.plot(x, results_MA.fittedvalues, label='MA')
# plt.plot(results_ARIMA.fittedvalues, label='ARIMA')
plt.legend()

plt.figure()
plt.plot(x, norway_temperature, label="Original")
plt.plot(x, results_AR.fittedvalues, label='AR')
plt.plot(x, results_MA.fittedvalues, label='MA')
# plt.plot(results_ARIMA.fittedvalues, label='ARIMA')
plt.xlim(1910, 1920)
plt.legend()

prediction = results_AR.predict(0, number_of_months_to_predict)
# forecast = results_AR.forecast(0, 7)
plt.figure()
plt.plot(x, norway_temperature, label="Original", color="blue")
plt.plot(p, prediction, label="prediction", color="yellow")
plt.plot(p, -prediction, label="-prediction", color="red")
plt.xlim(2000, 2012+(number_of_months_to_predict+1)/12)
plt.legend()

plt.show()
