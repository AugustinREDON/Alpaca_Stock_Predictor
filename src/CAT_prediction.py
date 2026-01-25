# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def main():
    # Load environment variables from .env
    load_dotenv()

    API_KEY = os.getenv("ALPACA_API_KEY")
    API_SECRET = os.getenv("ALPACA_API_SECRET")

    if not API_KEY or not API_SECRET:
        raise ValueError(
            "Missing Alpaca credentials. Ensure .env has your ALPACA_API_KEY and ALPACA_API_SECRET."
        )

    # Stock and date range
    SYMBOL = "SPY"
    START_DATE = "2022-01-01"
    END_DATE = "2026-01-22"

    # Pull daily bars from Alpaca
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    request_params = StockBarsRequest( #bar includes open, high, low, close, volume
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Day,
        start=START_DATE,
        end=END_DATE,
    )

    bars = client.get_stock_bars(request_params)

    # Convert to DataFrame (multi-index); reset for easier handling
    df = bars.df.reset_index()

    # Filter just in case multiple symbols are returned
    if "symbol" in df.columns:
        df = df[df["symbol"] == SYMBOL].copy()

    # Ensure chronological order
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    # Feature engineering
    df["return"] = df["close"].pct_change() #daily return (closetoday - closeyesterday)/closeyesterday
    df["volatility"] = df["return"].rolling(5).std()  #5-day rolling stddev of returns
    df["ma_5"] = df["close"].rolling(5).mean() 
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # Target: next day's close
    df["target"] = df["close"].shift(-1)

    # Drop rows made invalid by rolling windows / shift
    df = df.dropna()

    features = ["close", "volume", "volatility", "ma_5", "ma_10", "ma_20"]
    X = df[features]
    y = df["target"]

    # Train/test split (shuffle=False because time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error: ${mae:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label="Actual Price")
    plt.plot(preds, label="Predicted Price")
    plt.title(f"{SYMBOL} Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Next-day prediction from latest row
    latest_data = X.iloc[-1].values
    next_close = model.predict([latest_data])[0]
    print(f"Next Day Predicted Close for {SYMBOL}: ${next_close:.2f}")


if __name__ == "__main__":
    main()
