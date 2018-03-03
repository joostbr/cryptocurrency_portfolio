import pandas as pd


# Compute the Bollinger Bands
def BollingerBands(data, window=50):
    MA = data["close"].rolling(window=window).mean()
    SD = data["close"].rolling(window=window).std()
    data['upperBB'] = MA + (2 * SD)
    data['lowerBB'] = MA - (2 * SD)
    return data


def CCI(data, ndays=10):
    TP = (data['high'] + data['low'] + data['close']) / 3
    CCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)),
                    name='CCI')
    data = data.join(CCI)
    return data


# Ease of Movement
def EVM(data, ndays):
    meanvol = data['volume'].mean()

    dm = ((data['high'] + data['low']) / 2) - ((data['high'].shift(1) + data['low'].shift(1)) / 2)
    br = (data['volume'] / 10000000000) / ((data['high'] - data['low']))
    EVM = dm / br
    EVM_MA = pd.Series(pd.rolling_mean(EVM, ndays), name='EVM')
    data = data.join(EVM_MA)
    return data
