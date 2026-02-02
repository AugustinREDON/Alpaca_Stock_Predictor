# Alpaca Stock Predictor

## Overview

This project is a **Python-based machine learning application** that **predicts next-day stock returns**, which are reconstructed into price estimates using historical market data from the **Alpaca API** and a **Random Forest Regressor**.

Instead of forecasting raw prices directly, the model predicts **next-day returns**, which are then converted into price estimates. This design improves stability and avoids common extrapolation issues in tree-based models.

> Disclaimer: This project is for learning/demo purposes only and is not financial advice.

## What the Project Does

- Pulls **daily historical stock data (OHLCV)** from Alpaca  
- Engineers technical features:
  - Rolling volatility  
  - Moving-average distance ratios (5 / 10 / 20 days)  
  - Trading volume  
- Trains a **RandomForestRegressor** to predict **next-day returns**  
- Converts predicted returns into **next-day price estimates**  
- Evaluates performance using **Mean Absolute Error (MAE)**  
- Visualizes **actual vs predicted prices**  
- Outputs a **next-day price prediction** for the most recent trading day  

---

## Modeling Approach (High-Level)

### Inputs (Time t)
- Volume  
- 5-day rolling volatility  
- Distance from 5-day, 10-day, and 20-day moving averages  

### Target
- Return at time **t + 1**

### Price Reconstruction

<div align="center">

**Predicted Price = Price<sub>t</sub> × (1 + Predicted Return)**

</div>

- Uses a time-aware train/test split (`shuffle=False`) to prevent data leakage
- Model: Random Forest Regressor (300 trees, depth-limited)

##  Why Predict Returns Instead of Prices?
The original version of this project attempted to predict raw prices, which led to flatlining behavior when prices exceeded levels seen in the training data.

Because Random Forests cannot extrapolate beyond observed values, predictions clustered around historical price ranges.

### Solution
The model was redesigned to predict returns instead of prices:
- Returns are scale-invariant
- Patterns generalize better across different price regimes
- Results are more stable and interpretable

## Known Limitations
The model is designed as a one-step predictor, using information available at time t to estimate the return at t + 1. As a result, predictions are inherently lagged, since the model can only react once new information is reflected in the data. By operating on returns rather than raw prices, the model works with a signal that is closer to stationary, which helps it self-correct over time as new observations enter the feature window. However, this also means that during sudden regime shifts or high-impact events, adjustments occur with a slight delay. The model relies exclusively on technical price-based features and does not incorporate macroeconomic indicators, fundamentals, or news data, and it is therefore intended for analysis and experimentation rather than live trading or execution.

## Setup

To run the project, you’ll need a free Alpaca account.  
Create one at https://alpaca.markets and generate API credentials from the dashboard. Then run the following commands:

```bash
git clone https://github.com/AugustinREDON/Alpaca_Stock_Predictor.git
cd Alpaca_Stock_Predictor
pip install -r requirements.txt
```
Create a `.env` file in the project root:
```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_secret_key_here
```