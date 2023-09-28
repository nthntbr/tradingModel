import numpy
import requests
import json
from pandas import DataFrame
from sklearn import linear_model
import csv


def extract_values(data):
    
    # Extract the values from the inner dictionary and convert them to float
    values_list = [float(value) for value in data.values()]
    
    return values_list



def dataCategoryOne(data):
    
    newData = []
    for x in data['Time Series (Daily)']:
        openPrice = float(data['Time Series (Daily)'][x]['1. open'])
        high = float(data['Time Series (Daily)'][x]['2. high'])
        low = float(data['Time Series (Daily)'][x]['3. low'])
        closePrice = float(data['Time Series (Daily)'][x]['4. close'])
        volume = int(data['Time Series (Daily)'][x]['5. volume'])
        normalized_value = (closePrice - high) / (high -low)
        target = 0
        if((closePrice - openPrice) > 0):
            target = 1
        newData.append([target, volume, normalized_value])

    count = 0
    final_data = []
    for x in newData[:-21]:
        day = [x[0], x[1]]
        closing_prices = [[sublist[2]] for sublist in newData if isinstance(sublist, list) and sublist][(count + 1):(count + 21)]
        count += 1
        i = list(range(1, 21))
        i.reverse()
        regr = linear_model.LinearRegression()
        regr.fit(closing_prices, i)
        day.append(regr.coef_[0])
        regr.fit(closing_prices[0:9], i[0:9])
        day.append(regr.coef_[0])
        regr.fit(closing_prices[0:4], i[0:4])
        day.append(regr.coef_[0])
        regr.fit(closing_prices[0:2], i[0:2])
        day.append(regr.coef_[0])
        final_data.append(day)

    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['target', 'volume', 'normalized value', '20-day', '10-day', '5-day', '3-day'])
        writer.writerows(final_data)

def compute_rsi(data, window=14):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.rolling(window=window).mean()
    roll_down = down.abs().rolling(window=window).mean()

    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))

    return RSI

def dataCategoryTwo(data):

    df = DataFrame(data["Time Series (Daily)"]).T
    df['1. open'] = df['1. open'].astype(float)
    df['2. high'] = df['2. high'].astype(float)
    df['3. low'] = df['3. low'].astype(float)
    df['4. close'] = df['4. close'].astype(float)
    df['5. volume'] = df['5. volume'].astype(int)
    df = df.iloc[::-1].reset_index(drop=True)
    df['Return'] = df['4. close'].pct_change()

    df['Lagged_Return'] = df['Return'].shift(1)

    df['5_Day_MA'] = df['4. close'].rolling(window=5).mean()

    df['RSI'] = compute_rsi(df['4. close'])


    df['Next_Day_Return'] = df['4. close'].pct_change().shift(-1)
    df['Target'] = numpy.where(df['Next_Day_Return'] > 0, 1, 0)

    df.to_csv('data.csv', index=False)



url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=XGG6PGRW3V07P3E5'
r = requests.get(url)
data = r.json()
dataCategoryOne(data)
dataCategoryTwo(data)


