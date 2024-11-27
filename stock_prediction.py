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
    Train the prediction model
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

def plot_predictions(df, predictions, last_n_days=30):
    """
    Plot actual vs predicted prices
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-last_n_days:], df['Close'][-last_n_days:], label='Actual Price', color='blue')
    plt.plot(df.index[-last_n_days:], predictions[-last_n_days:], label='Predicted Price', color='red', linestyle='--')
    plt.title('S&P 500 - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Get S&P 500 data
    print("Fetching S&P 500 data...")
    df = get_stock_data()
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Train model
    print("\nTraining model...")
    model, train_score, test_score = train_model(X, y)
    
    print(f"\nModel Performance:")
    print(f"Training Score: {train_score:.4f}")
    print(f"Testing Score: {test_score:.4f}")
    
    # Make predictions
    predictions = model.predict(X)
    
    # Plot results
    plot_predictions(df, predictions)
    
    # Predict next day
    last_data = X.iloc[-1:].values
    next_day_pred = model.predict(last_data)[0]
    last_close = df['Close'].iloc[-1]
    
    print(f"\nLast Close: ${last_close:.2f}")
    print(f"Next Day Prediction: ${next_day_pred:.2f}")
    print(f"Predicted Change: {((next_day_pred - last_close) / last_close * 100):.2f}%")

if __name__ == "__main__":
    main() 