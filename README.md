# Next-Day Price Predictor (Alpaca + Random Forest)

A small end-to-end Python project that pulls historical daily data for the stock of yoru choice from the **Alpaca Market Data API**, engineers basic technical features (returns, rolling volatility, moving averages), trains a **RandomForestRegressor**, and predicts the **next trading day’s closing price**. The script reports **Mean Absolute Error (MAE)** on a holdout period and plots **actual vs. predicted** prices.

> Disclaimer: This project is for learning/demo purposes only and is not financial advice.

---

## What it does

- Fetches daily bars (**open, high, low, close, volume**) from Alpaca for:
  - `SYMBOL = "Stock name"`
  - `START_DATE = "2022-01-01"`
  - `END_DATE = "2026-01-22"`
- Feature engineering:
  - Daily return: `close.pct_change()`
  - 5-day rolling volatility: rolling std. dev. of returns
  - Moving averages: 5 / 10 / 20 day
- Target:
  - Predicts **next day close** (`target = close.shift(-1)`)
- Trains a **Random Forest** model (time-series friendly split: `shuffle=False`)
- Outputs:
  - MAE (in dollars)
  - Plot: actual vs predicted on the test window
  - Next-day predicted close using the most recent feature row

---

## Tech stack

- Python
- Alpaca Market Data API (`alpaca-py`)
- scikit-learn (RandomForestRegressor)
- pandas / numpy
- matplotlib
- python-dotenv

---



