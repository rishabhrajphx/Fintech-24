import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def get_stock_data(ticker, start, end):
    """
    Fetch stock data using yfinance
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}.")
    return df

def prepare_data(df):
    """
    Prepare data for prediction
    """
    # Create features
    df['MA5'] = df['Close'].shift(1).rolling(window=5).mean()
    df['MA20'] = df['Close'].shift(1).rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'].shift(1))
    
    # Create target (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI']
    X = df[features]
    y = df['Target']

    return X, y

def calculate_rsi(data, periods=14):
    """
    Calculate RSI technical indicator
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def train_model(X, y):
    """
    Train the linear regression model to predict stock prices.
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Calculate accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    return model, train_score, test_score
