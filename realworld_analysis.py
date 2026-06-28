"""
Real-World Data Analysis — End-to-End Pipeline
================================================
Domain 1: FINANCE  — Stock performance, returns, risk metrics, portfolio analysis
Domain 2: HEALTH   — Patient risk profiling, CVD prediction, clinical insights
Domain 3: RETAIL   — Revenue analysis, seasonality, store performance, forecasting
"""

import json, warnings
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, kruskal, chi2_contingency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
warnings.filterwarnings("ignore")

report = {}

print("=" * 65)
print("  REAL-WORLD DATA ANALYSIS PIPELINE")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 1 : FINANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n📈  FINANCE ANALYSIS")
print("-" * 45)

stock = pd.read_csv("/home/claude/finance_stocks.csv", parse_dates=["date"])
tickers = stock["ticker"].unique()

finance_results = {}
for tkr in tickers:
    df = stock[stock["ticker"] == tkr].copy()

    # Returns
    daily_ret  = df["daily_ret"].dropna()
    annual_ret = (1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe     = (annual_ret - 0.06) / annual_vol   # rf = 6%
    max_dd     = ((df["close"] / df["close"].cummax()) - 1).min()

    # VaR & CVaR (95%)
    var_95  = np.percentile(daily_ret, 5)
    cvar_95 = daily_ret[daily_ret <= var_95].mean()

    # Beta (vs equal-weight market)
    mkt = stock.groupby("date")["daily_ret"].mean().reset_index()
    merged = df.merge(mkt, on="date", suffixes=("","_mkt")).dropna()
    if len(merged) > 50:
        beta = np.cov(merged["daily_ret"], merged["daily_ret_mkt"])[0,1] / \
               np.var(merged["daily_ret_mkt"])
    else:
        beta = 1.0

    finance_results[tkr] = {
        "start_price":  round(float(df["close"].iloc[0]), 2),
        "end_price":    round(float(df["close"].iloc[-1]), 2),
        "total_return": round(float((df["close"].iloc[-1]/df["close"].iloc[0]-1)*100), 2),
        "annual_return":round(float(annual_ret*100), 2),
        "annual_vol":   round(float(annual_vol*100), 2),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd*100), 2),
        "var_95":       round(float(var_95*100), 4),
        "cvar_95":      round(float(cvar_95*100), 4),
        "beta":         round(float(beta), 4),
    }

    print(f"  {tkr:8s}  Return={annual_ret*100:.1f}%  Sharpe={sharpe:.2f}  "
          f"MaxDD={max_dd*100:.1f}%  Beta={beta:.2f}")

# Correlation matrix between tickers
pivot   = stock.pivot_table(index="date", columns="ticker", values="daily_ret")
corr_mx = pivot.corr().round(4)

report["finance"] = {
    "tickers":      finance_results,
    "corr_matrix":  corr_mx.to_dict(),
}
print(f"\n  Stock correlation matrix computed ({len(tickers)} tickers)")

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 2 : HEALTH
# ══════════════════════════════════════════════════════════════════════════════
print("\n🏥  HEALTH ANALYSIS")
print("-" * 45)

health = pd.read_csv("/home/claude/health_patients.csv")
print(f"  Patients: {len(health):,}  |  CVD rate: {health['cvd_event'].mean()*100:.1f}%")

# Clinical statistics by outcome
cvd_yes = health[health["cvd_event"]==1]
cvd_no  = health[health["cvd_event"]==0]

clinical_cols = ["age","bmi","systolic_bp","cholesterol","glucose",
                 "hba1c","exercise_hrs_wk","heart_rate"]
clinical_stats = {}
for col in clinical_cols:
    stat, p = stats.mannwhitneyu(cvd_yes[col], cvd_no[col], alternative="two-sided")
    clinical_stats[col] = {
        "cvd_mean":    round(float(cvd_yes[col].mean()), 2),
        "no_cvd_mean": round(float(cvd_no[col].mean()), 2),
        "difference":  round(float(cvd_yes[col].mean() - cvd_no[col].mean()), 2),
        "mwu_p":       round(float(p), 6),
        "significant": bool(p < 0.05),
    }
    sig = "✅" if p < 0.05 else "❌"
    print(f"  {col:20s}  CVD={cvd_yes[col].mean():.1f}  No-CVD={cvd_no[col].mean():.1f}  {sig}")

# Risk factor prevalence
risk_factors = {}
for factor in ["smoker","diabetic","hypertension","obesity","family_hx_cvd"]:
    overall_rate = health[factor].mean() * 100
    cvd_rate     = cvd_yes[factor].mean() * 100
    no_cvd_rate  = cvd_no[factor].mean() * 100
    # Chi-square
    ct = pd.crosstab(health[factor], health["cvd_event"])
    chi2, p, _, _ = chi2_contingency(ct)
    risk_factors[factor] = {
        "overall_pct":  round(overall_rate, 1),
        "cvd_pct":      round(cvd_rate, 1),
        "no_cvd_pct":   round(no_cvd_rate, 1),
        "chi2":         round(float(chi2), 4),
        "p_value":      round(float(p), 6),
    }

# CVD Prediction Model
FEAT_COLS = ["age","bmi","systolic_bp","cholesterol","glucose","hba1c",
             "exercise_hrs_wk","heart_rate","smoker","diabetic",
             "hypertension","obesity","family_hx_cvd","alcohol_units_wk"]

X = health[FEAT_COLS]
y = health["cvd_event"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            random_state=42, stratify=y)

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                      ("clf", LogisticRegression(max_iter=1000, C=0.5, random_state=42))]),
    "Random Forest":        Pipeline([("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))]),
    "Gradient Boosting":    Pipeline([("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42))]),
}

health_model_results = {}
best_model, best_auc = None, 0
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:,1]
    auc     = roc_auc_score(y_te, y_proba)
    f1      = f1_score(y_te, y_pred)
    acc     = accuracy_score(y_te, y_pred)
    cv_auc  = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc")

    health_model_results[name] = {
        "auc": round(auc,4), "f1": round(f1,4), "accuracy": round(acc,4),
        "cv_auc_mean": round(cv_auc.mean(),4), "cv_auc_std": round(cv_auc.std(),4),
        "y_pred": y_pred.tolist(), "y_proba": y_proba.tolist(),
    }
    if auc > best_auc:
        best_auc = auc; best_model = name
    print(f"  {name:25s}  AUC={auc:.4f}  F1={f1:.4f}  CV-AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}")

# Feature importance from RF
rf_pipe = models["Random Forest"]
rf_imp  = pd.DataFrame({
    "feature":    FEAT_COLS,
    "importance": rf_pipe.named_steps["clf"].feature_importances_
}).sort_values("importance", ascending=False)

print(f"\n  Best Model: {best_model} (AUC={best_auc:.4f})")

report["health"] = {
    "overview":      {"n_patients": len(health), "cvd_rate": round(health["cvd_event"].mean()*100,2)},
    "clinical_stats": clinical_stats,
    "risk_factors":   risk_factors,
    "models":         health_model_results,
    "feature_importance": rf_imp.to_dict("records"),
    "y_test":         y_te.tolist(),
    "best_model":     best_model,
}

# ══════════════════════════════════════════════════════════════════════════════
#  DOMAIN 3 : RETAIL
# ══════════════════════════════════════════════════════════════════════════════
print("\n🛒  RETAIL ANALYSIS")
print("-" * 45)

retail = pd.read_csv("/home/claude/retail_sales.csv", parse_dates=["date"])
print(f"  Rows: {len(retail):,}  |  Stores: {retail['store_id'].nunique()}  |  "
      f"Revenue: ₹{retail['revenue'].sum()/1e9:.2f}B")

# Year-over-year growth
yearly = retail.groupby("year")["revenue"].sum()
yoy    = {}
for yr in [2023, 2024]:
    if yr in yearly.index and (yr-1) in yearly.index:
        growth = (yearly[yr] / yearly[yr-1] - 1) * 100
        yoy[str(yr)] = round(float(growth), 2)
        print(f"  YoY Growth {yr}: {growth:+.1f}%")

# Category performance
cat_perf = retail.groupby("category").agg(
    total_revenue=("revenue","sum"),
    total_profit=("profit","sum"),
    avg_margin=("margin_pct","mean"),
    total_units=("units_sold","sum")
).round(2).sort_values("total_revenue", ascending=False)

# Store tier performance
tier_perf = retail.groupby("store_tier").agg(
    total_revenue=("revenue","sum"),
    n_stores=("store_id","nunique"),
    avg_daily_rev=("revenue","mean")
).round(2)

# Seasonality: monthly revenue index
monthly_idx = retail.groupby("month")["revenue"].mean()
monthly_idx = (monthly_idx / monthly_idx.mean() * 100).round(2)

# Weekend vs Weekday
wkday_wkend = retail.groupby("is_weekend")["revenue"].agg(["mean","sum"]).round(2)
wkend_lift  = round(float((wkday_wkend.loc[1,"mean"] / wkday_wkend.loc[0,"mean"] - 1)*100), 2)
print(f"  Weekend revenue lift: +{wkend_lift:.1f}% vs weekday")

# Store rankings
store_rank = retail.groupby("store_id")["revenue"].sum().sort_values(ascending=False)

# Sales forecasting (Linear Regression on monthly data)
monthly_sales = retail.groupby(["year","month"])["revenue"].sum().reset_index()
monthly_sales["time_idx"] = range(len(monthly_sales))
X_f = sm.add_constant(monthly_sales[["time_idx","month"]])
y_f = monthly_sales["revenue"]
ols = sm.OLS(y_f, X_f).fit()

report["retail"] = {
    "overview": {
        "total_revenue_B": round(float(retail["revenue"].sum()/1e9), 3),
        "total_profit_B":  round(float(retail["profit"].sum()/1e9), 3),
        "n_stores": int(retail["store_id"].nunique()),
        "yoy_growth": yoy,
    },
    "category_performance": cat_perf.reset_index().to_dict("records"),
    "store_tier_performance": tier_perf.reset_index().to_dict("records"),
    "monthly_seasonality_index": monthly_idx.to_dict(),
    "weekend_lift_pct": wkend_lift,
    "store_rankings": store_rank.to_dict(),
    "forecast_r2": round(float(ols.rsquared), 4),
}

retail.groupby(["date","category"])["revenue"].sum().reset_index().to_csv(
    "/home/claude/retail_daily_cat.csv", index=False)

print(f"  OLS Revenue Forecast R² = {ols.rsquared:.4f}")
print(f"\n  Top 3 Stores by Revenue:")
for s, v in store_rank.head(3).items():
    print(f"    {s}: ₹{v/1e6:.1f}M")

with open("/home/claude/realworld_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# Save feature importance
rf_imp.to_csv("/home/claude/health_feature_importance.csv", index=False)

print("\n✅ realworld_report.json saved")
print("✅ health_feature_importance.csv saved")
print("✅ retail_daily_cat.csv saved")
