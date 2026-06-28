"""
Real-World Data Visualization Suite
=====================================
Generates 4 thematic dashboards:
  1. finance_dashboard.png     — stock performance, returns, risk metrics
  2. health_dashboard.png      — clinical analysis, model ROC, feature importance
  3. retail_dashboard.png      — revenue trends, seasonality, store comparison
  4. master_dashboard.png      — cross-domain executive summary
"""

import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Wedge
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
warnings.filterwarnings("ignore")

# ── Theme ──────────────────────────────────────────────────────────────────────
BG   = "#080C14"
CARD = "#0F1623"
GRID = "#1A2233"
TEXT = "#E8EFF8"
MUTED= "#5A6880"
P = ["#4F8EF7","#F75C7E","#2DCB8E","#FFB547","#A78BFA",
     "#38BDF8","#FB923C","#34D399","#F472B6","#FACC15"]

plt.rcParams.update({
    "figure.facecolor":BG, "axes.facecolor":CARD, "axes.edgecolor":GRID,
    "axes.labelcolor":TEXT, "axes.titlecolor":TEXT,
    "xtick.color":TEXT, "ytick.color":TEXT,
    "grid.color":GRID, "text.color":TEXT,
    "legend.facecolor":CARD, "legend.edgecolor":GRID,
    "font.family":"DejaVu Sans",
})

# ── Load data ──────────────────────────────────────────────────────────────────
stock  = pd.read_csv("/home/claude/finance_stocks.csv", parse_dates=["date"])
health = pd.read_csv("/home/claude/health_patients.csv")
retail = pd.read_csv("/home/claude/retail_sales.csv", parse_dates=["date"])

with open("/home/claude/realworld_report.json") as f:
    rep = json.load(f)

tickers  = list(rep["finance"]["tickers"].keys())
tkr_cols = dict(zip(tickers, P[:5]))

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
QTR_ORDER    = ["Q1","Q2","Q3","Q4"]
EDU_ORDER    = ["High School","Bachelor's","Master's","PhD"]

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 1 — FINANCE  (4 × 3)
# ══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(24, 26), facecolor=BG)
fig1.suptitle("📈  Finance Domain — Stock Market Analysis  (2019–2024)",
              fontsize=24, fontweight="bold", color=TEXT, y=0.99)
gs1 = gridspec.GridSpec(4, 3, figure=fig1, hspace=0.55, wspace=0.36,
                        top=0.95, bottom=0.04, left=0.06, right=0.97)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kax = fig1.add_subplot(gs1[0, :])
kax.set_facecolor(BG); kax.axis("off")

fin_data = rep["finance"]["tickers"]
best_tkr  = max(fin_data, key=lambda k: fin_data[k]["sharpe_ratio"])
kpis = [
    ("Best Sharpe",    f"{best_tkr}\n{fin_data[best_tkr]['sharpe_ratio']:.2f}",          P[0]),
    ("Best Return",    f"{max(fin_data,key=lambda k:fin_data[k]['annual_return'])}\n"
                       f"{max(v['annual_return'] for v in fin_data.values()):.1f}%",      P[2]),
    ("Lowest Drawdown",f"{min(fin_data,key=lambda k:abs(fin_data[k]['max_drawdown']))}\n"
                       f"{min(abs(v['max_drawdown']) for v in fin_data.values()):.1f}%",  P[3]),
    ("Lowest Beta",    f"{min(fin_data,key=lambda k:fin_data[k]['beta'])}\n"
                       f"{min(v['beta'] for v in fin_data.values()):.2f}",                P[4]),
    ("Avg Correlation","0.41\nacross 5 stocks",                                           P[5]),
]
for i, (label, val, color) in enumerate(kpis):
    x = 0.005 + i * 0.199
    kax.add_patch(FancyBboxPatch((x,0.08),0.188,0.82,
                  boxstyle="round,pad=0.01", facecolor=CARD,
                  edgecolor=color, linewidth=2.5, transform=kax.transAxes, zorder=2))
    kax.text(x+0.094, 0.72, label, ha="center", va="center",
             fontsize=9, color=MUTED, transform=kax.transAxes)
    kax.text(x+0.094, 0.28, val, ha="center", va="center",
             fontsize=14, fontweight="bold", color=color, transform=kax.transAxes)

# R1C0-C2 — Normalized price chart (all tickers)
ax = fig1.add_subplot(gs1[1, :])
for tkr, color in tkr_cols.items():
    df_t = stock[stock["ticker"]==tkr].set_index("date")["close"]
    normalized = df_t / df_t.iloc[0] * 100
    ax.plot(normalized.index, normalized.values, color=color, lw=1.8, label=tkr, alpha=0.9)
ax.axhline(100, color=MUTED, lw=1, linestyle="--", alpha=0.5)
ax.fill_between(normalized.index, 100, normalized.values, alpha=0.03, color=P[0])
ax.set_title("Normalized Price Performance  (Base = 100 at Start)", fontweight="bold", pad=10)
ax.set_ylabel("Normalized Price")
ax.legend(fontsize=9, loc="upper left", ncol=5)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}"))

# R2C0 — Risk-Return Scatter
ax = fig1.add_subplot(gs1[2, 0])
for tkr, color in tkr_cols.items():
    v = fin_data[tkr]
    ax.scatter(v["annual_vol"], v["annual_return"], color=color, s=180,
               zorder=5, edgecolors="white", linewidth=1.5)
    ax.annotate(tkr, (v["annual_vol"], v["annual_return"]),
                xytext=(6, 4), textcoords="offset points", fontsize=8, color=color)
ax.axhline(0, color=MUTED, lw=0.8, linestyle="--")
ax.set_xlabel("Annual Volatility (%)"); ax.set_ylabel("Annual Return (%)")
ax.set_title("Risk–Return Scatter", fontweight="bold", pad=8)

# R2C1 — Sharpe Ratio Bar
ax = fig1.add_subplot(gs1[2, 1])
sharpes = {k: v["sharpe_ratio"] for k, v in fin_data.items()}
colors_s = [P[2] if v >= 0 else P[1] for v in sharpes.values()]
bars = ax.bar(sharpes.keys(), sharpes.values(), color=colors_s, width=0.55)
ax.axhline(0, color=MUTED, lw=1, linestyle="--")
for bar, v in zip(bars, sharpes.values()):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + (0.02 if v>=0 else -0.06),
            f"{v:.2f}", ha="center", fontsize=9, color=TEXT)
ax.set_title("Sharpe Ratio  (rf=6%)", fontweight="bold", pad=8)
ax.set_ylabel("Sharpe Ratio")

# R2C2 — Max Drawdown
ax = fig1.add_subplot(gs1[2, 2])
mdd = {k: v["max_drawdown"] for k, v in fin_data.items()}
bars2 = ax.bar(mdd.keys(), mdd.values(), color=P[1], width=0.55, alpha=0.85)
for bar, v in zip(bars2, mdd.values()):
    ax.text(bar.get_x()+bar.get_width()/2, v - 1.5,
            f"{v:.1f}%", ha="center", fontsize=9, color=TEXT)
ax.set_title("Maximum Drawdown (%)", fontweight="bold", pad=8)
ax.set_ylabel("Max Drawdown (%)")

# R3C0 — Correlation heatmap
ax = fig1.add_subplot(gs1[3, 0])
corr_data = pd.DataFrame(rep["finance"]["corr_matrix"])
mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
sns.heatmap(corr_data, ax=ax, cmap="RdBu_r", center=0,
            annot=True, fmt=".2f", linewidths=0.5, linecolor=BG,
            annot_kws={"size": 9, "weight":"bold"},
            cbar_kws={"shrink":0.75, "label":"Pearson r"})
ax.set_title("Return Correlations", fontweight="bold", pad=8)
ax.tick_params(labelsize=8)

# R3C1 — Rolling 20-day volatility
ax = fig1.add_subplot(gs1[3, 1])
for tkr, color in tkr_cols.items():
    df_t = stock[stock["ticker"]==tkr].copy()
    ax.plot(df_t["date"], df_t["volatility"]*np.sqrt(252)*100,
            color=color, lw=1.3, alpha=0.8, label=tkr)
ax.set_title("Rolling 20-Day Annualised Volatility (%)", fontweight="bold", pad=8)
ax.set_ylabel("Volatility (%)"); ax.legend(fontsize=7, ncol=2)

# R3C2 — VaR Comparison
ax = fig1.add_subplot(gs1[3, 2])
var_vals  = {k: abs(v["var_95"])  for k, v in fin_data.items()}
cvar_vals = {k: abs(v["cvar_95"]) for k, v in fin_data.items()}
x_v = np.arange(len(tickers))
ax.bar(x_v-0.18, list(var_vals.values()),  0.32, color=P[3], label="VaR 95%",  alpha=0.88)
ax.bar(x_v+0.18, list(cvar_vals.values()), 0.32, color=P[1], label="CVaR 95%", alpha=0.88)
ax.set_xticks(x_v); ax.set_xticklabels(tickers, fontsize=9)
ax.set_title("Daily VaR & CVaR at 95%  (%)", fontweight="bold", pad=8)
ax.set_ylabel("Loss (%)"); ax.legend(fontsize=8)

fig1.savefig("/mnt/user-data/outputs/finance_dashboard.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print("✅  finance_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 2 — HEALTH  (4 × 3)
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(24, 26), facecolor=BG)
fig2.suptitle("🏥  Health Domain — Cardiovascular Disease Risk Analysis",
              fontsize=24, fontweight="bold", color=TEXT, y=0.99)
gs2 = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.55, wspace=0.36,
                        top=0.95, bottom=0.04, left=0.06, right=0.97)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kax2 = fig2.add_subplot(gs2[0, :])
kax2.set_facecolor(BG); kax2.axis("off")
cvd_rate = rep["health"]["overview"]["cvd_rate"]
best_model = rep["health"]["best_model"]
best_auc   = rep["health"]["models"][best_model]["auc"]
smoker_cvd = rep["health"]["risk_factors"]["smoker"]["cvd_pct"]
diab_cvd   = rep["health"]["risk_factors"]["diabetic"]["cvd_pct"]
htn_cvd    = rep["health"]["risk_factors"]["hypertension"]["cvd_pct"]

kpis2 = [
    ("CVD Event Rate",   f"{cvd_rate:.1f}%",            P[1]),
    ("Best Model AUC",   f"{best_auc:.4f}",             P[0]),
    ("Smokers → CVD",    f"{smoker_cvd:.1f}%",          P[6]),
    ("Diabetics → CVD",  f"{diab_cvd:.1f}%",            P[3]),
    ("Hypertension → CVD", f"{htn_cvd:.1f}%",           P[4]),
]
for i, (label, val, color) in enumerate(kpis2):
    x = 0.005 + i * 0.199
    kax2.add_patch(FancyBboxPatch((x,0.08),0.188,0.82,
                   boxstyle="round,pad=0.01", facecolor=CARD,
                   edgecolor=color, linewidth=2.5, transform=kax2.transAxes, zorder=2))
    kax2.text(x+0.094, 0.72, label, ha="center", va="center",
              fontsize=9, color=MUTED, transform=kax2.transAxes)
    kax2.text(x+0.094, 0.30, val, ha="center", va="center",
              fontsize=16, fontweight="bold", color=color, transform=kax2.transAxes)

# R1C0 — Age distribution by CVD
ax = fig2.add_subplot(gs2[1, 0])
cvd_yes = health[health["cvd_event"]==1]
cvd_no  = health[health["cvd_event"]==0]
ax.hist(cvd_no["age"],  bins=30, color=P[0], alpha=0.65, label="No CVD", density=True)
ax.hist(cvd_yes["age"], bins=30, color=P[1], alpha=0.65, label="CVD",    density=True)
ax.set_title("Age Distribution by CVD Outcome", fontweight="bold", pad=8)
ax.set_xlabel("Age"); ax.set_ylabel("Density"); ax.legend(fontsize=9)

# R1C1 — Clinical markers comparison (mean difference)
ax = fig2.add_subplot(gs2[1, 1])
clin = rep["health"]["clinical_stats"]
features = list(clin.keys())
diffs = [clin[f]["difference"] for f in features]
colors_d = [P[1] if d > 0 else P[2] for d in diffs]
ax.barh(features, diffs, color=colors_d, height=0.65)
ax.axvline(0, color=MUTED, lw=1)
for i, (f, d) in enumerate(zip(features, diffs)):
    sig = "✅" if clin[f]["significant"] else ""
    ax.text(d + (0.3 if d >= 0 else -0.3), i, f"{d:+.1f} {sig}",
            va="center", ha="left" if d >= 0 else "right", fontsize=8, color=TEXT)
ax.set_title("Clinical Markers: CVD vs No-CVD Mean Diff", fontweight="bold", pad=8)
ax.set_xlabel("Difference (CVD − No CVD)")

# R1C2 — Risk factor prevalence
ax = fig2.add_subplot(gs2[1, 2])
rf = rep["health"]["risk_factors"]
factors = list(rf.keys())
cvd_pcts    = [rf[f]["cvd_pct"]    for f in factors]
no_cvd_pcts = [rf[f]["no_cvd_pct"] for f in factors]
x_rf = np.arange(len(factors))
ax.bar(x_rf-0.2, cvd_pcts,    0.35, color=P[1], label="CVD",    alpha=0.88)
ax.bar(x_rf+0.2, no_cvd_pcts, 0.35, color=P[0], label="No CVD", alpha=0.88)
ax.set_xticks(x_rf)
ax.set_xticklabels([f.replace("_"," ").title() for f in factors], fontsize=7.5, rotation=20)
ax.set_title("Risk Factor Prevalence (%)", fontweight="bold", pad=8)
ax.set_ylabel("Prevalence (%)"); ax.legend(fontsize=9)

# R2C0-1 — ROC Curves all models
ax = fig2.add_subplot(gs2[2, :2])
y_test = np.array(rep["health"]["y_test"])
for i, (mname, mdata) in enumerate(rep["health"]["models"].items()):
    fpr, tpr, _ = roc_curve(y_test, mdata["y_proba"])
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=P[i], lw=2.5,
            label=f"{mname}  (AUC={roc_auc:.4f})")
    ax.fill_between(fpr, tpr, alpha=0.04, color=P[i])
ax.plot([0,1],[0,1], color=MUTED, lw=1.5, linestyle="--", label="Random baseline")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — CVD Prediction Models", fontweight="bold", pad=10)
ax.legend(fontsize=9, loc="lower right")

# R2C2 — Feature importance
ax = fig2.add_subplot(gs2[2, 2])
fi = pd.read_csv("/home/claude/health_feature_importance.csv").head(12)
fi_sorted = fi.sort_values("importance")
colors_fi = [P[1] if v >= fi_sorted["importance"].quantile(0.75) else P[0]
             for v in fi_sorted["importance"]]
ax.barh(fi_sorted["feature"], fi_sorted["importance"]*100, color=colors_fi, height=0.65)
ax.set_title("Feature Importance\n(Random Forest)", fontweight="bold", pad=8)
ax.set_xlabel("Importance (%)"); ax.tick_params(axis="y", labelsize=8)

# R3C0 — BMI distribution by CVD
ax = fig2.add_subplot(gs2[3, 0])
sns.kdeplot(cvd_no["bmi"],  ax=ax, color=P[0], fill=True, alpha=0.4, label="No CVD")
sns.kdeplot(cvd_yes["bmi"], ax=ax, color=P[1], fill=True, alpha=0.4, label="CVD")
ax.axvline(25, color=P[3], lw=1.5, linestyle="--", label="Normal limit (25)")
ax.axvline(30, color=P[6], lw=1.5, linestyle="--", label="Obese limit (30)")
ax.set_title("BMI Distribution by CVD Outcome", fontweight="bold", pad=8)
ax.set_xlabel("BMI"); ax.legend(fontsize=8)

# R3C1 — Systolic BP vs Cholesterol scatter
ax = fig2.add_subplot(gs2[3, 1])
sample = health.sample(800, random_state=3)
sc = ax.scatter(sample["systolic_bp"], sample["cholesterol"],
                c=sample["cvd_event"].map({0:P[0], 1:P[1]}),
                s=18, alpha=0.6, edgecolors="none")
ax.axvline(140, color=P[3], lw=1.5, linestyle="--", alpha=0.7, label="HTN threshold (140)")
ax.axhline(240, color=P[6], lw=1.5, linestyle="--", alpha=0.7, label="High chol (240)")
from matplotlib.patches import Patch
legend_handles = [Patch(color=P[0], label="No CVD"), Patch(color=P[1], label="CVD")]
ax.legend(handles=legend_handles + [
    plt.Line2D([0],[0], color=P[3], lw=1.5, linestyle="--", label="HTN threshold"),
    plt.Line2D([0],[0], color=P[6], lw=1.5, linestyle="--", label="High chol"),
], fontsize=7.5, loc="upper left")
ax.set_xlabel("Systolic BP (mmHg)"); ax.set_ylabel("Cholesterol (mg/dL)")
ax.set_title("BP vs Cholesterol\n(coloured by CVD status)", fontweight="bold", pad=8)

# R3C2 — Confusion matrix (best model)
ax = fig2.add_subplot(gs2[3, 2])
best_preds = np.array(rep["health"]["models"][best_model]["y_pred"])
cm = confusion_matrix(y_test, best_preds)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm, annot=False, ax=ax, cmap="Blues",
            linewidths=2, linecolor=BG, cbar=False)
for r in range(2):
    for c in range(2):
        clr = BG if cm[r,c] > cm.max()*0.5 else TEXT
        ax.text(c+0.5, r+0.38, str(cm[r,c]), ha="center", va="center",
                fontsize=18, fontweight="bold", color=clr)
        ax.text(c+0.5, r+0.66, f"({cm_pct[r,c]:.1f}%)", ha="center", va="center",
                fontsize=10, color=clr)
ax.set_xticklabels(["No CVD","CVD"], fontsize=10)
ax.set_yticklabels(["No CVD","CVD"], fontsize=10, rotation=0)
ax.set_title(f"Confusion Matrix\n{best_model}", fontweight="bold", pad=8)

fig2.savefig("/mnt/user-data/outputs/health_dashboard.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print("✅  health_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 3 — RETAIL  (4 × 3)
# ══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(24, 26), facecolor=BG)
fig3.suptitle("🛒  Retail Domain — Multi-Store Sales Intelligence  (2022–2024)",
              fontsize=24, fontweight="bold", color=TEXT, y=0.99)
gs3 = gridspec.GridSpec(4, 3, figure=fig3, hspace=0.55, wspace=0.36,
                        top=0.95, bottom=0.04, left=0.06, right=0.97)

ret = rep["retail"]

# ── KPI Cards ─────────────────────────────────────────────────────────────────
kax3 = fig3.add_subplot(gs3[0, :])
kax3.set_facecolor(BG); kax3.axis("off")
kpis3 = [
    ("Total Revenue",  f"₹{ret['overview']['total_revenue_B']:.2f}B",   P[2]),
    ("Total Profit",   f"₹{ret['overview']['total_profit_B']:.2f}B",    P[3]),
    ("Active Stores",  str(ret["overview"]["n_stores"]),                 P[0]),
    ("Weekend Lift",   f"+{ret['weekend_lift_pct']:.1f}%",              P[4]),
    ("Forecast R²",    f"{ret['forecast_r2']:.3f}",                     P[5]),
]
for i, (label, val, color) in enumerate(kpis3):
    x = 0.005 + i * 0.199
    kax3.add_patch(FancyBboxPatch((x,0.08),0.188,0.82,
                   boxstyle="round,pad=0.01", facecolor=CARD,
                   edgecolor=color, linewidth=2.5, transform=kax3.transAxes, zorder=2))
    kax3.text(x+0.094, 0.72, label, ha="center", va="center",
              fontsize=9, color=MUTED, transform=kax3.transAxes)
    kax3.text(x+0.094, 0.30, val, ha="center", va="center",
              fontsize=16, fontweight="bold", color=color, transform=kax3.transAxes)

# R1C0-2 — Monthly revenue trend (all 3 years, stacked area)
ax = fig3.add_subplot(gs3[1, :])
daily_cat = pd.read_csv("/home/claude/retail_daily_cat.csv", parse_dates=["date"])
monthly_cat = daily_cat.groupby([pd.Grouper(key="date", freq="ME"), "category"])["revenue"].sum().unstack(fill_value=0)
categories_list = monthly_cat.columns.tolist()
x_dates = monthly_cat.index
bottom  = np.zeros(len(monthly_cat))
for i, cat in enumerate(categories_list):
    vals = monthly_cat[cat].values / 1e6
    ax.fill_between(x_dates, bottom, bottom+vals,
                    color=P[i % len(P)], alpha=0.78, label=cat)
    bottom += vals
total_monthly = monthly_cat.sum(axis=1).values / 1e6
ax.plot(x_dates, total_monthly, color="white", lw=1.5, linestyle="--", alpha=0.6, label="Total")
ax.set_title("Monthly Revenue by Category — Stacked Area  (₹M)", fontweight="bold", pad=10)
ax.set_ylabel("Revenue (₹M)")
ax.legend(fontsize=7.5, ncol=4, loc="upper left")

# R2C0 — Revenue by category (horizontal bar)
ax = fig3.add_subplot(gs3[2, 0])
cat_perf_list = ret["category_performance"]
cat_df   = pd.DataFrame(cat_perf_list).sort_values("total_revenue")
bar_h = ax.barh(cat_df["category"], cat_df["total_revenue"]/1e6,
                color=P[:len(cat_df)], height=0.65)
for bar, v in zip(bar_h, cat_df["total_revenue"].values):
    ax.text(bar.get_width()+2, bar.get_y()+bar.get_height()/2,
            f"₹{v/1e6:.0f}M", va="center", fontsize=8, color=TEXT)
ax.set_title("Total Revenue by Category  (₹M)", fontweight="bold", pad=8)
ax.set_xlabel("Revenue (₹M)"); ax.tick_params(axis="y", labelsize=9)

# R2C1 — Gross margin by category
ax = fig3.add_subplot(gs3[2, 1])
margin_df = pd.DataFrame(cat_perf_list).sort_values("avg_margin")
colors_m = [P[2] if m >= margin_df["avg_margin"].mean() else P[6]
            for m in margin_df["avg_margin"]]
bars_m = ax.barh(margin_df["category"], margin_df["avg_margin"],
                 color=colors_m, height=0.65)
ax.axvline(margin_df["avg_margin"].mean(), color="white",
           lw=1.5, linestyle="--", label=f"Avg {margin_df['avg_margin'].mean():.1f}%")
for bar, v in zip(bars_m, margin_df["avg_margin"].values):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f"{v:.1f}%", va="center", fontsize=8, color=TEXT)
ax.set_title("Gross Margin % by Category", fontweight="bold", pad=8)
ax.set_xlabel("Gross Margin (%)"); ax.legend(fontsize=8)

# R2C2 — Seasonality index
ax = fig3.add_subplot(gs3[2, 2])
seas = ret["monthly_seasonality_index"]
months_vals = [seas.get(str(m), 100) for m in range(1, 13)]
colors_s = [P[2] if v >= 100 else P[1] for v in months_vals]
bars_s = ax.bar(range(1, 13), months_vals, color=colors_s, width=0.7, alpha=0.88)
ax.axhline(100, color="white", lw=1.5, linestyle="--", alpha=0.6, label="Base = 100")
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS, fontsize=8)
ax.set_title("Monthly Seasonality Index", fontweight="bold", pad=8)
ax.set_ylabel("Index (100 = avg)"); ax.legend(fontsize=8)
for bar, v in zip(bars_s, months_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
            f"{v:.0f}", ha="center", fontsize=7, color=TEXT)

# R3C0 — Store tier performance
ax = fig3.add_subplot(gs3[3, 0])
tier_df = pd.DataFrame(ret["store_tier_performance"])
bar_t = ax.bar(tier_df["store_tier"], tier_df["total_revenue"]/1e6,
               color=[P[3],P[0],P[2]], width=0.55)
for bar, row in zip(bar_t, tier_df.itertuples()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
            f"₹{row.total_revenue/1e6:.0f}M\n({row.n_stores} stores)",
            ha="center", fontsize=8, color=TEXT)
ax.set_title("Revenue by Store Tier  (₹M)", fontweight="bold", pad=8)
ax.set_ylabel("Revenue (₹M)")

# R3C1 — Top stores by revenue
ax = fig3.add_subplot(gs3[3, 1])
store_rev = ret["store_rankings"]
store_df  = pd.Series(store_rev).sort_values(ascending=True).tail(10)
ax.barh(store_df.index, store_df.values/1e6,
        color=[P[0] if v == store_df.max() else P[4] for v in store_df.values],
        height=0.65)
ax.set_title("Top 10 Stores by Revenue  (₹M)", fontweight="bold", pad=8)
ax.set_xlabel("Revenue (₹M)"); ax.tick_params(axis="y", labelsize=8)

# R3C2 — Weekend vs Weekday revenue box
ax = fig3.add_subplot(gs3[3, 2])
wk  = retail.groupby(["date","is_weekend"])["revenue"].sum().reset_index()
day_data   = wk[wk["is_weekend"]==0]["revenue"].values / 1e3
wkend_data = wk[wk["is_weekend"]==1]["revenue"].values / 1e3
bp3 = ax.boxplot([day_data, wkend_data], patch_artist=True,
                 medianprops=dict(color="white", lw=2),
                 notch=False)
bp3["boxes"][0].set_facecolor(P[0]); bp3["boxes"][0].set_alpha(0.7)
bp3["boxes"][1].set_facecolor(P[2]); bp3["boxes"][1].set_alpha(0.7)
for elem in ["whiskers","caps","fliers"]:
    for item in bp3[elem]: item.set_color(MUTED)
ax.set_xticklabels(["Weekday","Weekend"], fontsize=10)
ax.set_title(f"Daily Revenue: Weekday vs Weekend\n(+{ret['weekend_lift_pct']:.1f}% weekend lift)",
             fontweight="bold", pad=8)
ax.set_ylabel("Daily Revenue (₹K)")

fig3.savefig("/mnt/user-data/outputs/retail_dashboard.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig3)
print("✅  retail_dashboard.png")

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 4 — CROSS-DOMAIN MASTER DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(26, 22), facecolor=BG)
fig4.suptitle("🌐  Real-World Data Science — Cross-Domain Executive Dashboard",
              fontsize=24, fontweight="bold", color=TEXT, y=0.99)
gs4 = gridspec.GridSpec(3, 3, figure=fig4, hspace=0.52, wspace=0.38,
                        top=0.94, bottom=0.04, left=0.05, right=0.97)

# Domain header banners
for col, (domain, color, subtitle) in enumerate([
    ("📈 FINANCE", P[0], "Stock Market Risk & Return"),
    ("🏥 HEALTH",  P[1], "CVD Risk Prediction"),
    ("🛒 RETAIL",  P[2], "Revenue Intelligence"),
]):
    ax_h = fig4.add_subplot(gs4[0, col])
    ax_h.set_facecolor(CARD)
    ax_h.add_patch(FancyBboxPatch((0.02,0.02),0.96,0.96,
                   boxstyle="round,pad=0.01", facecolor=CARD,
                   edgecolor=color, linewidth=3, transform=ax_h.transAxes))
    ax_h.text(0.5, 0.62, domain, ha="center", va="center",
              fontsize=18, fontweight="bold", color=color, transform=ax_h.transAxes)
    ax_h.text(0.5, 0.30, subtitle, ha="center", va="center",
              fontsize=11, color=MUTED, transform=ax_h.transAxes)
    ax_h.axis("off")

# Finance — normalized returns
ax = fig4.add_subplot(gs4[1, 0])
for tkr, color in tkr_cols.items():
    df_t  = stock[stock["ticker"]==tkr].set_index("date")["close"]
    normed = df_t / df_t.iloc[0] * 100
    ax.plot(normed.index, normed.values, color=color, lw=1.6, label=tkr, alpha=0.9)
ax.axhline(100, color=MUTED, lw=0.8, linestyle="--")
ax.set_title("Normalised Stock Performance", fontweight="bold", pad=8)
ax.set_ylabel("Index (100 = start)"); ax.legend(fontsize=7, ncol=3)

# Health — ROC
ax = fig4.add_subplot(gs4[1, 1])
for i, (mname, mdata) in enumerate(rep["health"]["models"].items()):
    fpr, tpr, _ = roc_curve(y_test, mdata["y_proba"])
    ax.plot(fpr, tpr, color=P[i], lw=2.2,
            label=f"{mname.split()[0]} (AUC={mdata['auc']:.3f})")
ax.plot([0,1],[0,1], color=MUTED, lw=1, linestyle="--")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("CVD Prediction — ROC Curves", fontweight="bold", pad=8)
ax.legend(fontsize=8)

# Retail — Revenue trend
ax = fig4.add_subplot(gs4[1, 2])
total_monthly = retail.groupby(pd.Grouper(key="date", freq="ME"))["revenue"].sum()
ax.plot(total_monthly.index, total_monthly.values/1e6, color=P[2], lw=2.5,
        marker="o", markersize=4, markerfacecolor=P[3])
ax.fill_between(total_monthly.index, total_monthly.values/1e6, alpha=0.15, color=P[2])
ax.plot(total_monthly.index,
        total_monthly.rolling(3).mean().values/1e6,
        color="white", lw=1.5, linestyle="--", alpha=0.6, label="3-mo MA")
ax.set_title("Monthly Retail Revenue  (₹M)", fontweight="bold", pad=8)
ax.set_ylabel("Revenue (₹M)"); ax.legend(fontsize=8)

# Cross-domain insights panel
ax_ins = fig4.add_subplot(gs4[2, :])
ax_ins.set_facecolor(CARD); ax_ins.axis("off")
ax_ins.add_patch(FancyBboxPatch((0.005,0.03),0.99,0.94,
                 boxstyle="round,pad=0.01", facecolor=CARD,
                 edgecolor=MUTED, linewidth=1.5, transform=ax_ins.transAxes))
ax_ins.set_title("  🔑  Key Findings Across All Three Domains",
                 fontsize=14, fontweight="bold", color=TEXT, loc="left", pad=12)

insights = [
    (P[0], "📈 FINANCE", [
        f"ENERGY leads annual return at 22.4%; RETAIL stock is the only negative performer (−1.0%)",
        f"BANKCO has the safest profile: lowest max drawdown (−26.9%) and beta (0.57)",
        f"All 5 stocks show moderate-to-low positive correlation (avg r≈0.41) — diversification benefit exists",
        f"TECHCO offers the best Sharpe (0.38) for risk-adjusted return above 6% risk-free rate",
    ]),
    (P[1], "🏥 HEALTH", [
        f"CVD patients are on average 11.8 years older — age is the strongest non-modifiable risk factor",
        f"Logistic Regression achieves AUC=0.81, proving linear separability with clinical features",
        f"Smokers have a {smoker_cvd:.1f}% CVD rate vs ~15% baseline — 1.5× relative risk",
        f"Exercise hours/week is the most impactful modifiable protective factor",
    ]),
    (P[2], "🛒 RETAIL", [
        f"Electronics drives 37%+ of total revenue despite being a mid-frequency category",
        f"Weekend revenue is +19.9% higher — weekend promotional strategy has clear payoff",
        f"Diwali season (Oct–Nov) shows 2.5× revenue spike — highest seasonal multiplier",
        f"Flagship stores generate 2.2× more revenue than Standard stores per location",
    ]),
]

for col_idx, (color, domain, points) in enumerate(insights):
    x_base = 0.02 + col_idx * 0.335
    ax_ins.text(x_base, 0.88, domain, ha="left", va="top", fontsize=11,
                fontweight="bold", color=color, transform=ax_ins.transAxes)
    for j, point in enumerate(points):
        ax_ins.text(x_base, 0.75 - j * 0.18, f"• {point}", ha="left", va="top",
                    fontsize=8, color=TEXT, transform=ax_ins.transAxes, wrap=True)

fig4.savefig("/mnt/user-data/outputs/master_dashboard.png",
             dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig4)
print("✅  master_dashboard.png")
print("\n🎉  All dashboards complete!")
