"""
Real-World Data Generator
==========================
Generates THREE domain-specific datasets:
  1. FINANCE    — 5-year daily stock price data (5 companies, OHLCV + indicators)
  2. HEALTH     — Patient clinical records (2,000 patients, diagnosis prediction)
  3. RETAIL     — 3-year sales data (multi-store, multi-SKU, seasonal patterns)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(2024)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 1: FINANCE — Stock Market Data
# ══════════════════════════════════════════════════════════════════════════════
def generate_stock_prices(ticker, start_price, volatility, drift, n_days):
    """Geometric Brownian Motion for realistic stock simulation."""
    dt      = 1 / 252
    returns = np.random.normal(drift * dt, volatility * np.sqrt(dt), n_days)
    prices  = [start_price]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    return np.array(prices[1:])

tickers = {
    "TECHCO": {"price": 150,  "vol": 0.28, "drift": 0.18},
    "PHARMA": {"price": 85,   "vol": 0.22, "drift": 0.12},
    "ENERGY": {"price": 60,   "vol": 0.35, "drift": 0.08},
    "BANKCO": {"price": 120,  "vol": 0.20, "drift": 0.10},
    "RETAIL": {"price": 200,  "vol": 0.25, "drift": 0.15},
}

biz_days  = pd.bdate_range("2019-01-01", "2024-12-31")
n_days    = len(biz_days)
stock_dfs = []

for ticker, params in tickers.items():
    close = generate_stock_prices(ticker, params["price"], params["vol"], params["drift"], n_days)

    # OHLCV generation
    gap   = np.random.uniform(-0.015, 0.015, n_days)
    open_ = close * (1 + gap)
    high  = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.008, n_days)))
    low   = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.008, n_days)))
    vol   = np.random.lognormal(14 + np.random.uniform(-1, 1), 0.4, n_days).astype(int)

    df = pd.DataFrame({
        "date":   biz_days,
        "ticker": ticker,
        "open":   open_.round(2),
        "high":   high.round(2),
        "low":    low.round(2),
        "close":  close.round(2),
        "volume": vol,
    })

    # Technical indicators
    df["ma_20"]      = df["close"].rolling(20).mean().round(4)
    df["ma_50"]      = df["close"].rolling(50).mean().round(4)
    df["ma_200"]     = df["close"].rolling(200).mean().round(4)
    df["daily_ret"]  = df["close"].pct_change().round(6)
    df["volatility"] = df["daily_ret"].rolling(20).std().round(6)

    # RSI (14-day)
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).round(2)

    # MACD
    ema12        = df["close"].ewm(span=12).mean()
    ema26        = df["close"].ewm(span=26).mean()
    df["macd"]   = (ema12 - ema26).round(4)
    df["signal"] = df["macd"].ewm(span=9).mean().round(4)

    # Bollinger Bands
    df["bb_mid"]   = df["close"].rolling(20).mean().round(4)
    bb_std         = df["close"].rolling(20).std()
    df["bb_upper"] = (df["bb_mid"] + 2 * bb_std).round(4)
    df["bb_lower"] = (df["bb_mid"] - 2 * bb_std).round(4)

    # Market cap proxy
    shares_out = np.random.randint(500, 2000) * 1_000_000
    df["mkt_cap_M"] = (df["close"] * shares_out / 1e6).round(0)

    stock_dfs.append(df)

stock_df = pd.concat(stock_dfs, ignore_index=True)
stock_df.to_csv("/home/claude/finance_stocks.csv", index=False)
print(f"✅ Finance dataset  : {stock_df.shape[0]:,} rows × {stock_df.shape[1]} cols")

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 2: HEALTH — Patient Clinical Records
# ══════════════════════════════════════════════════════════════════════════════
N = 2500

age      = np.random.normal(52, 14, N).clip(18, 90).astype(int)
gender   = np.random.choice(["Male","Female"], N, p=[0.48, 0.52])
bmi      = np.random.normal(27.5, 5.5, N).clip(15, 50).round(1)
smoker   = (np.random.rand(N) < (0.25 + (age - 40) * 0.003)).astype(int).clip(0,1)
diabetic = (bmi > 30) & (np.random.rand(N) < 0.35) | (np.random.rand(N) < 0.05)
diabetic = diabetic.astype(int)

# Vitals (correlated with age, bmi, smoking)
systolic_bp  = (115 + age * 0.45 + bmi * 0.6 + smoker * 8
                + np.random.normal(0, 10, N)).clip(90, 200).astype(int)
diastolic_bp = (systolic_bp * np.random.uniform(0.55, 0.70, N)).astype(int)
heart_rate   = (72 + smoker * 5 - (age - 40) * 0.1
                + np.random.normal(0, 8, N)).clip(45, 130).astype(int)
cholesterol  = (180 + age * 0.8 + bmi * 1.2 + smoker * 20 + diabetic * 25
                + np.random.normal(0, 25, N)).clip(100, 350).astype(int)
glucose      = (90 + diabetic * 55 + bmi * 0.8
                + np.random.normal(0, 15, N)).clip(60, 400).astype(int)
hba1c        = (5.2 + diabetic * 1.8 + bmi * 0.04
                + np.random.normal(0, 0.4, N)).clip(4.5, 12).round(1)

# Lab results
creatinine = (0.9 + age * 0.005 + np.random.normal(0, 0.2, N)).clip(0.4, 4.0).round(2)
wbc_count  = np.random.normal(7.0, 1.5, N).clip(2.5, 15).round(1)
hemoglobin = np.random.normal(13.5, 1.8, N).clip(8, 18).round(1)

# Exercise & lifestyle
exercise_hrs_wk = (3.5 - age * 0.03 + np.random.normal(0, 1.5, N)).clip(0, 14).round(1)
alcohol_units_wk= np.random.exponential(3, N).clip(0, 30).round(1)
stress_level    = np.random.choice(["Low","Medium","High"], N, p=[0.30, 0.45, 0.25])

# Comorbidities
hypertension  = (systolic_bp > 140).astype(int)
obesity       = (bmi >= 30).astype(int)
family_hx_cvd = (np.random.rand(N) < 0.30).astype(int)

# Primary outcome: Cardiovascular Disease risk (logistic)
risk_score = (
    -6.0
    + 0.06  * age
    + 0.04  * systolic_bp
    + 0.008 * cholesterol
    + 1.2   * smoker
    + 0.8   * diabetic
    + 0.05  * bmi
    - 0.15  * exercise_hrs_wk
    + 0.6   * family_hx_cvd
    + np.random.normal(0, 0.4, N)
)
cvd_risk_prob = 1 / (1 + np.exp(-risk_score))
cvd_event     = (np.random.rand(N) < cvd_risk_prob).astype(int)

health_df = pd.DataFrame({
    "patient_id":      [f"P{i:05d}" for i in range(1, N+1)],
    "age":             age,
    "gender":          gender,
    "bmi":             bmi,
    "smoker":          smoker,
    "diabetic":        diabetic,
    "hypertension":    hypertension,
    "obesity":         obesity,
    "family_hx_cvd":   family_hx_cvd,
    "systolic_bp":     systolic_bp,
    "diastolic_bp":    diastolic_bp,
    "heart_rate":      heart_rate,
    "cholesterol":     cholesterol,
    "glucose":         glucose,
    "hba1c":           hba1c,
    "creatinine":      creatinine,
    "wbc_count":       wbc_count,
    "hemoglobin":      hemoglobin,
    "exercise_hrs_wk": exercise_hrs_wk,
    "alcohol_units_wk":alcohol_units_wk,
    "stress_level":    stress_level,
    "cvd_event":       cvd_event,
    "cvd_risk_prob":   cvd_risk_prob.round(4),
})

health_df.to_csv("/home/claude/health_patients.csv", index=False)
print(f"✅ Health dataset   : {health_df.shape[0]:,} rows × {health_df.shape[1]} cols  |  CVD rate: {cvd_event.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 3: RETAIL — Multi-Store Sales Data
# ══════════════════════════════════════════════════════════════════════════════
stores     = [f"STORE_{i:02d}" for i in range(1, 16)]   # 15 stores
categories = {
    "Electronics":   {"base": 18000, "margin": 0.18, "seasonal": "flat"},
    "Clothing":      {"base": 4500,  "margin": 0.55, "seasonal": "winter"},
    "Groceries":     {"base": 2200,  "margin": 0.22, "seasonal": "flat"},
    "Sports":        {"base": 6500,  "margin": 0.42, "seasonal": "summer"},
    "Home Decor":    {"base": 9500,  "margin": 0.48, "seasonal": "diwali"},
    "Books":         {"base": 1200,  "margin": 0.35, "seasonal": "flat"},
    "Toys":          {"base": 3800,  "margin": 0.45, "seasonal": "christmas"},
}

store_tiers = {s: np.random.choice(["Flagship","Standard","Mini"],
               p=[0.20, 0.50, 0.30]) for s in stores}
store_multiplier = {"Flagship": 2.2, "Standard": 1.0, "Mini": 0.55}

dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
retail_rows = []

for date in dates:
    month     = date.month
    is_weekend= date.weekday() >= 5
    # Seasonal indices
    season_factor = {
        "flat":      1.0,
        "winter":    1.0 + 0.5 * np.cos((month - 12) * np.pi / 3) if month in [11,12,1,2] else 0.85,
        "summer":    1.0 + 0.4 * np.sin((month - 3) * np.pi / 6) if month in [4,5,6,7] else 0.80,
        "diwali":    2.5 if month in [10, 11] else 0.90,
        "christmas": 2.2 if month == 12 else 0.80,
    }

    for store in stores:
        tier_mult = store_multiplier[store_tiers[store]]
        for cat, params in categories.items():
            sf    = season_factor.get(params["seasonal"], 1.0)
            base  = params["base"] * tier_mult * sf
            noise = np.random.normal(1.0, 0.12)
            wkend = 1.25 if is_weekend else 1.0
            units = max(1, int(np.random.poisson(max(1, base / params["base"] * 8)) * wkend))
            price = params["base"] * np.random.uniform(0.85, 1.15)
            revenue = units * price * noise
            cogs    = revenue * (1 - params["margin"])
            profit  = revenue - cogs

            retail_rows.append({
                "date":       date,
                "store_id":   store,
                "store_tier": store_tiers[store],
                "category":   cat,
                "units_sold": units,
                "unit_price": round(price, 2),
                "revenue":    round(revenue, 2),
                "cogs":       round(cogs, 2),
                "profit":     round(profit, 2),
                "margin_pct": round(params["margin"] * 100, 1),
                "is_weekend": int(is_weekend),
                "month":      month,
                "quarter":    f"Q{(month-1)//3 + 1}",
                "year":       date.year,
                "day_of_week":date.strftime("%A"),
            })

retail_df = pd.DataFrame(retail_rows)
retail_df.to_csv("/home/claude/retail_sales.csv", index=False)
print(f"✅ Retail dataset   : {retail_df.shape[0]:,} rows × {retail_df.shape[1]} cols")
print(f"   Revenue total    : ₹{retail_df['revenue'].sum()/1e9:.2f}B")
