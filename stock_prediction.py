import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(symbol='^GSPC', period='2y'):
    """
    Fetch stock data using yfinance
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def prepare_data(df):
    """
    Prepare data for prediction
    """
    # Create features
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Create target (next day's closing price)
    df['Target'] = df['Close'].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features
    features = ['Close', 'Volume', 'MA5', 'MA20', 'RSI']
    X = df[features]
    y = df['Target']
    
    return X, y

