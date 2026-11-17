"""Used car regression: should the dealer buy this car at auction?

Reproduces every number in the slides. Run:  python3 analysis.py
Outputs go to ./output (model summaries, metrics, auction decisions, charts).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

SEED = 42
TEST_FRAC = 0.30
PREP_COST = 1_500

FULL_FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident) + Num_Owners"
REDUCED_FORMULA = "Sale_Price ~ Mileage_K + Age_Years + Horsepower + C(Accident)"


def money(x):
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def rmse(actual, pred):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def r2(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    return float(1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2))


# ---------------------------------------------------------------- 1. Split
sales = pd.read_csv(DATA / "used_car_sales.csv")
auction = pd.read_csv(DATA / "auction_lots.csv")

rng = np.random.default_rng(SEED)
test_idx = rng.choice(sales.index, size=round(TEST_FRAC * len(sales)), replace=False)
test = sales.loc[test_idx].reset_index(drop=True)
train = sales.drop(index=test_idx).reset_index(drop=True)

# ---------------------------------------------------------------- 2. Fit
full = smf.ols(FULL_FORMULA, data=train).fit()
reduced = smf.ols(REDUCED_FORMULA, data=train).fit()

# Num_Owners is not significant (p ~ 0.23) and dropping it leaves adjusted R² essentially
# unchanged (0.8802 -> 0.8799), so the simpler model is the one handed over as v1.
v1, v1_name = reduced, "reduced (no Num_Owners)"

# ---------------------------------------------------------------- 3. Evaluate on test cars
baseline_price = train["Sale_Price"].mean()
test_pred_v1 = v1.predict(test)
test_pred_base = np.full(len(test), baseline_price)

fit_table = pd.DataFrame(
    {
        "R2_test": [r2(test.Sale_Price, test_pred_base), r2(test.Sale_Price, test_pred_v1)],
        "RMSE_test_$": [rmse(test.Sale_Price, test_pred_base), rmse(test.Sale_Price, test_pred_v1)],
    },
    index=["Baseline (train average)", "v1 regression"],
)


# ---------------------------------------------------------------- 4. Decision rules
def v1_rule(car: dict, auction_price: float) -> str:
    """Buy if auction price + prep cost is below v1's predicted sale price."""
    predicted = float(v1.predict(pd.DataFrame([car])).iloc[0])
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def baseline_rule(car: dict, auction_price: float) -> str:
    """Buy if auction price + prep cost is below the training-set average sale price."""
    return "buy" if auction_price + PREP_COST < baseline_price else "pass"


features = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
auction["Pred_v1"] = v1.predict(auction)
auction["Pred_baseline"] = baseline_price
auction["Profit_if_bought"] = auction.Later_Sale_Price - auction.Auction_Price - PREP_COST
for name, rule in [("v1", v1_rule), ("baseline", baseline_rule)]:
    auction[f"Decision_{name}"] = [
        rule(row[features].to_dict(), row.Auction_Price) for _, row in auction.iterrows()
    ]


def summarize(decision_col):
    bought = auction[auction[decision_col] == "buy"]
    return {
        "Cars bought": len(bought),
        "Money-losing cars bought": int((bought.Profit_if_bought < 0).sum()),
        "Total profit $": int(bought.Profit_if_bought.sum()),
        "Profit per car bought $": round(bought.Profit_if_bought.mean()) if len(bought) else 0,
    }


decision_table = pd.DataFrame(
    {"Baseline rule": summarize("Decision_baseline"), "v1 rule": summarize("Decision_v1")}
)
# Reference point: perfect hindsight (buy exactly the profitable cars)
hindsight = auction[auction.Profit_if_bought > 0]

# ---------------------------------------------------------------- 5. Save outputs
(OUT / "model_full.txt").write_text(str(full.summary()))
(OUT / "model_reduced.txt").write_text(str(reduced.summary()))
fit_table.round(3).to_csv(OUT / "test_fit_comparison.csv")
decision_table.to_csv(OUT / "auction_decision_comparison.csv")
auction.round(0).to_csv(OUT / "auction_decisions_by_lot.csv", index=False)

# Chart 1: predicted vs actual on test cars
fig, ax = plt.subplots(figsize=(6.5, 5.5))
lo, hi = 0, 50_000
ax.plot([lo, hi], [lo, hi], color="#999999", lw=1, ls="--", label="Perfect prediction")
ax.scatter(test.Sale_Price, test_pred_base, s=22, color="#bbbbbb", label="Baseline (average price)")
ax.scatter(test.Sale_Price, test_pred_v1, s=22, color="#1f4e79", label="Regression model (v1)")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Actual sale price ($)"); ax.set_ylabel("Predicted sale price ($)")
ax.set_title(f"Test cars the model never saw (n={len(test)})")
fmt = matplotlib.ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K")
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "pred_vs_actual_test.png", dpi=200); plt.close(fig)

# Chart 2: auction outcome comparison
fig, ax = plt.subplots(figsize=(6.5, 4))
labels = ["Baseline rule", "v1 rule"]
profits = [decision_table.loc["Total profit $", c] for c in labels]
bars = ax.bar(labels, profits, color=["#bbbbbb", "#1f4e79"], width=0.5)
for b, c in zip(bars, labels):
    n = decision_table.loc["Cars bought", c]; l = decision_table.loc["Money-losing cars bought", c]
    ax.annotate(f"{money(b.get_height())}\n{n} bought, {l} lose money",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom" if b.get_height() >= 0 else "top", fontsize=10)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Total profit on 40 auction lots ($)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: money(x)))
ax.spines[["top", "right"]].set_visible(False)
ymin, ymax = min(0, min(profits)), max(0, max(profits))
pad = 0.25 * (ymax - ymin)
ax.set_ylim(ymin - (pad if ymin < 0 else 0), ymax + pad)
fig.tight_layout(); fig.savefig(OUT / "auction_profit_comparison.png", dpi=200); plt.close(fig)

# ---------------------------------------------------------------- Print report
pd.set_option("display.width", 120)
print(f"Data: {len(sales)} cars -> train {len(train)} / test {len(test)} (seed {SEED})")
print(f"Training-set average sale price (baseline): ${baseline_price:,.0f}\n")
print("FULL MODEL\n", full.summary().tables[1])
print(f"Adj R² full = {full.rsquared_adj:.4f} | reduced = {reduced.rsquared_adj:.4f}")
print(f"v1 = {v1_name}\n")
print("V1 MODEL\n", v1.summary().tables[1], f"\nR² = {v1.rsquared:.4f}, adj R² = {v1.rsquared_adj:.4f}\n")
print("TEST-SET FIT\n", fit_table.round(3), "\n")
print("AUCTION DECISIONS (40 lots)\n", decision_table, "\n")
print(f"Hindsight best: {len(hindsight)} profitable cars, ${hindsight.Profit_if_bought.sum():,.0f}")
print("\nLots each rule buys:")
print(auction[["Lot", "Auction_Price", "Pred_v1", "Later_Sale_Price", "Profit_if_bought",
               "Decision_baseline", "Decision_v1"]].round(0).to_string(index=False))
