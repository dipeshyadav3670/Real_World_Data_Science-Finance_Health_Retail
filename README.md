# 🌐 Real-World Data Science — Finance · Health · Retail

A production-style, end-to-end data science project applied across three
real-world domains on a combined dataset of 125,000+ records. Each domain
tackles a distinct problem — quantitative risk modeling, clinical prediction,
and business revenue intelligence — demonstrating the full data science
workflow from raw data generation to actionable conclusions.

---

## 🗂️ Domains at a Glance

| Domain | Dataset | Problem Type | Key Result |
|---|---|---|---|
| 📈 Finance | 7,830 rows — 5 stocks × 5 years | Risk-return analysis | ENERGY: 22.4% annual return, Sharpe 0.47 |
| 🏥 Health | 2,500 patient records | CVD binary classification | Logistic Regression AUC = 0.8105 |
| 🛒 Retail | 115,080 transactions | Revenue analysis + forecasting | ₹6.02B revenue, +19.9% weekend lift |

---

## 📈 Domain 1 — Finance (Stock Market Analysis)

### What's Inside
- 5-year daily OHLCV data for 5 companies generated via
  Geometric Brownian Motion (industry-standard stochastic process)
- 12 technical indicators computed per ticker:
  MA(20/50/200), RSI(14), MACD + Signal Line, Bollinger Bands,
  daily returns, rolling 20-day volatility

### Analysis Performed
- Annualised return, volatility, and Sharpe Ratio (rf = 6%)
- Maximum Drawdown calculation using cumulative peak tracking
- Daily VaR (Value at Risk) and CVaR at 95% confidence
- Beta coefficient vs equal-weight market portfolio
- Full return correlation matrix across all 5 tickers

### Key Findings
- ENERGY leads raw returns (22.4%/yr) but carries highest beta (1.75)
- BANKCO is the defensive choice — lowest drawdown (−26.9%), lowest beta (0.57)
- RETAIL stock is the only loser (−1.0%) with worst drawdown (−57.7%)
- Average inter-stock correlation ~0.41 — meaningful diversification benefit
- VaR range: BANKCO −1.26%/day (safest) to ENERGY −2.18%/day (riskiest)

---

## 🏥 Domain 2 — Health (CVD Risk Prediction)

### What's Inside
- 2,500 patient records with 23 clinical and lifestyle features
- Realistic correlations: BMI drives diabetes, smoking amplifies BP,
  age compounds all risk factors
- CVD event rate: 17.9% (clinically realistic high-risk cohort)

### Analysis Performed
- Mann-Whitney U tests across 8 clinical markers (CVD vs No-CVD)
- Chi-square tests for 5 binary risk factors
- Three ML classifiers: Logistic Regression, Random Forest,
  Gradient Boosting with 5-fold cross-validation
- Feature importance via Random Forest (Gini impurity)
- ROC curves, confusion matrices, F1-score evaluation

### Key Findings
- CVD patients are on average 11.8 years older — age is the
  strongest non-modifiable risk factor
- Heart rate alone is NOT a significant differentiator (p = 0.18)
  — a common clinical misconception
- Logistic Regression wins (AUC=0.81) — relationship is largely
  linear in log-odds space, consistent with Framingham Risk Score
- Diabetics face 2.45× relative CVD risk vs non-diabetics
- Exercise hrs/week is the most impactful MODIFIABLE protective factor
- Top 5 features: Age > Systolic BP > Cholesterol > Glucose > BMI

---

## 🛒 Domain 3 — Retail (Multi-Store Revenue Intelligence)

### What's Inside
- 115,080 daily sales records: 15 stores × 7 categories × 3 years
- 3 store tiers: Flagship (3), Standard (7), Mini (5)
- Realistic seasonal multipliers:
  Diwali 2.5× | Christmas 2.2× | Summer sports 1.4×

### Analysis Performed
- Year-over-Year revenue growth analysis
- Category-level margin vs revenue matrix
- Monthly seasonality index (base 100 normalisation)
- Store tier benchmarking (revenue per location)
- Weekend vs weekday lift quantification
- OLS revenue forecasting (R² = 0.47)
- Store performance ranking across all 15 locations

### Key Findings
- Electronics drives 20.6% of revenue at only 18% gross margin
- Fashion + Home Decor offer the best margin-revenue balance
  (55% and 48% margins respectively)
- Weekend revenue is +19.9% higher than weekday average
- Diwali season (Oct–Nov) peaks at 150+ seasonality index
- Flagship stores generate 2.87× more revenue per location
  than Mini stores
- February is the weakest month (index = 84)

---

## 📁 Project Structure

├── generate_realworld_data.py    # All 3 dataset generators
├── realworld_analysis.py         # End-to-end analysis pipeline
├── realworld_visualizations.py   # 4 dashboard generators
├── RealWorld_Report.md           # Full structured findings report
│
├── finance_stocks.csv            # 7,830-row OHLCV dataset
├── health_patients.csv           # 2,500-row patient records
├── retail_sales.csv              # 115,080-row transactions
│
├── realworld_report.json         # Complete machine-readable results
├── health_feature_importance.csv # RF feature importance scores
│
├── master_dashboard.png          # Cross-domain executive summary
├── finance_dashboard.png         # 12-panel stock analysis
├── health_dashboard.png          # Clinical analysis + ROC curves
└── retail_dashboard.png          # Revenue + seasonality charts

---

## 📊 Dashboards

| Dashboard | Panels | Highlights |
|---|---|---|
| `master_dashboard.png` | 9 panels | Cross-domain KPI cards, key findings per domain |
| `finance_dashboard.png` | 12 panels | Normalised price chart, risk-return scatter, VaR bars, correlation heatmap |
| `health_dashboard.png` | 12 panels | ROC curves, feature importance, clinical marker diffs, confusion matrix |
| `retail_dashboard.png` | 12 panels | Stacked area revenue, seasonality index, tier comparison, weekend lift |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| Pandas | Data wrangling, aggregation, time-series grouping |
| NumPy | Numerical computation, GBM simulation, array ops |
| Matplotlib | Multi-panel figure composition, dark theme dashboards |
| Seaborn | Heatmaps, violin plots, KDE, statistical charts |
| Scikit-learn | ML pipeline, preprocessing, cross-validation, metrics |
| SciPy | Statistical tests (Mann-Whitney U, Chi-square) |
| Statsmodels | OLS regression, trend modeling |

---

## 🚀 Getting Started

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels

# Step 1 — Generate all three datasets
python generate_realworld_data.py

# Step 2 — Run end-to-end analysis
python realworld_analysis.py

# Step 3 — Generate all 4 dashboards
python realworld_visualizations.py

---

## 📌 Skills Demonstrated

- Quantitative Finance         → GBM, Sharpe, VaR, CVaR, Beta, Drawdown
- Clinical Data Science        → Medical feature engineering, AUC, ROC
- Retail Analytics             → Seasonality indexing, tier benchmarking
- Machine Learning             → Logistic Regression, Random Forest, GBM
- Statistical Testing          → Mann-Whitney U, Chi-square, OLS
- Data Visualization           → 4 multi-panel dark-theme dashboards
- End-to-End Pipeline Design   → Data → Analysis → Visualization → Report
-
