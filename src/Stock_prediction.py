# Import necessary libraries

import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime, timezone

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
    SYMBOL = (input("Enter stock symbol to predict (default: SPY): ") or "SPY").upper()
    START_DATE = "2022-01-01"
    #START_DATE = input("Enter start date (YYYY-MM-DD, default: all available): ").strip() or "1900-01-01"
    END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")

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
    df["next_close"] = df["close"].shift(-1)
    df["volatility"] = df["return"].rolling(5).std()  #5-day rolling stddev of returns

    # 3. Moving Averages (Relative distance instead of raw price)
    df["ma_5_ratio"] = df["close"] / df["close"].rolling(5).mean() - 1
    df["ma_10_ratio"] = df["close"] / df["close"].rolling(10).mean() - 1
    df["ma_20_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1

    # Target: next day's close
    #df["target"] = df["close"].shift(-1)
    df["target_return"] = df["return"].shift(-1)

    # Drop rows made invalid by rolling windows / shift
    df = df.dropna()

    features = ["volume", "volatility", "ma_5_ratio", "ma_10_ratio", "ma_20_ratio"]
    X = df[features]
    y = df["target_return"]

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
    # preds = model.predict(X_test)
    # mae = mean_absolute_error(y_test, preds)
    # print(f"Mean Absolute Error: ${mae:.2f}")

    # same as OG code
    pred_returns = model.predict(X_test)

    #this would be the calculated verison of the real date (not acurate)
    #actual_future_prices = base_prices * (1 + y_test)

    #this pulls the real data from df
    actual_future_prices = df.loc[y_test.index, "next_close"]

    #base prcice to numeric to calucalte next price
    base_prices = df.loc[y_test.index, "close"]
    #predicted price turned from mlutiplying real base of day before * predicted returns.
    predicted_future_prices = base_prices * (1 + pred_returns)

    mae = mean_absolute_error(actual_future_prices, predicted_future_prices)
    print(f"Mean Absolute Error: ${mae:.2f}")

    # 5. Plot the Prices
    plt.figure(figsize=(12, 6))

    #days_to_plot = len(y_test)
    days_to_plot = 10

    plt.plot(y_test.tail(days_to_plot).index,
             actual_future_prices.tail(days_to_plot).values,
             label="Actual Price", color='blue', alpha=0.6)

    plt.plot(predicted_future_prices.tail(days_to_plot).index,
             predicted_future_prices.tail(days_to_plot).values,
             label="Predicted Price", color='orange', linestyle='--')

    plt.title(f"{SYMBOL} Price Prediction (Reconstructed from Returns)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


    latest_data_df = X.iloc[[-1]]

    next_return = model.predict(latest_data_df)[0]

    current_price = df["close"].iloc[-1]
    next_price = current_price * (1 + next_return)

    print("------------------------------------------------")
    print(f"Latest Close ({df['timestamp'].iloc[-1].date()}): ${current_price:.2f}")
    print(f"Model Predicts Return: {next_return * 100:+.2f}%")
    print(f"Model Predicts Price:  ${next_price:.2f}")
    print("------------------------------------------------")


    plt.show()




if __name__ == "__main__":
    main()
