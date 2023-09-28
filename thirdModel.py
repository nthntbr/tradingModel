import pandas as pd
import talib
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def makeData():
    url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&outputsize=full&apikey=XGG6PGRW3V07P3E5'
    r = requests.get(url)
    data = r.json()
    # Load the data into a Pandas dataframe
    df = pd.DataFrame(data["Time Series (Daily)"]).T

    # Rename the columns for clarity
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)
    df['Volume'] = df['Volume'].astype(int)
    df = df.iloc[::-1].reset_index(drop=True)
    # Calculate the RSI, MACD, and ADX indicators for the past prices
    df['rsi'] = talib.RSI(df['Close'])
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['Close'])
    df['adx'] = talib.ADX(df['High'], df['Low'], df['Close'])

    # Label the market conditions based on the indicator values
    df['market_conditions_neutral'] = 0
    df['market_conditions_overbought'] = 0
    df['market_conditions_oversold'] = 0
    if(df['rsi'].all() > 70):
        df['market_conditions_overbought'] = 1
    elif(df['rsi'].all() < 30):
        df['market_conditions_oversold'] = 1
    else:
        df['market_conditions_neutral'] = 1
    


    df['market_conditions_trending'] = 0
    df.loc[(df['macd'] > df['macd_signal']) & (df['adx'] > 25), 'market_conditions_trending'] = 1
    df['price_direction'] = talib.LINEARREG_ANGLE(df['Close'])
    # Select the features for training the model
    df.to_csv('data.csv', index=False)

makeData()
df = pd.read_csv("data.csv") 

df.dropna(inplace=True)


# Feature columns (add new columns created from one-hot encoding)
features = ['market_conditions_overbought', 'market_conditions_oversold', 'market_conditions_trending', 'market_conditions_neutral']

X = df[features].astype(int)
X['Close'] = df['Close'].astype(float)

# Select the target variable
y = df['price_direction'].astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train a random forest classifier on the training data
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
# Test the model on the testing data
accuracy = clf.score(X_test, y_test)
print('Model accuracy:', accuracy)


# Use the trained model to make predictions on new data
new_data = [[100, 'bullish'], [50, 'bearish']]
predictions = clf.predict(new_data)
print('Predictions:', predictions)
