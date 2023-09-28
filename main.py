import numpy
import requests
import json
from pandas import DataFrame
from sklearn import linear_model
import csv

def getJsonReq(url):
    r = requests.get(url)
    return r.json()

def writeData(header, data):
    with open('data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

def translateJsonData(data):
    output = []
    for x in data['Time Series (Daily)']:
        openPrice = float(data['Time Series (Daily)'][x]['1. open'])
        high = float(data['Time Series (Daily)'][x]['2. high'])
        low = float(data['Time Series (Daily)'][x]['3. low'])
        closePrice = float(data['Time Series (Daily)'][x]['4. close'])
        volume = int(data['Time Series (Daily)'][x]['5. volume'])
        output.append([openPrice, high, low, closePrice, volume])

url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={sym}&outputsize={size}&apikey=XGG6PGRW3V07P3E5'.format(sym = 'IBM', size = 'full')
data = getJsonReq(url)

