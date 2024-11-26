import datetime

import pandas as pd
import yfinance as yahooFinance
from sklearn.preprocessing import StandardScaler

startDate = datetime.datetime(2022, 5, 31)
endDate = datetime.datetime(2024, 5, 31)

info = yahooFinance.Ticker("META")
pd.set_option("display.max_rows", None)

history = info.history(start=startDate, end=endDate)
history.to_csv("hist.csv")

df = pd.read_csv("hist.csv").drop_duplicates()

X = df.drop(["Dividends", "Stock Splits"], axis=1)
y = df["Date"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[["Volume", "High", "Low"]])
X_processed = pd.concat(
    [
        pd.DataFrame(
            X_scaled,
            columns=["Volume", "High", "Low"],
        ),
    ],
    axis=1,
)

print(X_processed)
