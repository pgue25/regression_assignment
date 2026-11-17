# regression_assignment

Used-car pricing model: can a dealer predict a car's lot sale price well enough to decide which cars to buy at auction?

## Run it

```bash
pip install -r requirements.txt
python3 analysis.py
```

`analysis.py` reproduces every number on the slides. The 70/30 split uses a fixed random seed (42), so results are identical on every run.

## Files

| Path | Contents |
|---|---|
| `data/used_car_sales.csv` | 240 cars the dealer sold in the last year (used to fit and test the model) |
| `data/auction_lots.csv` | 40 auction lots with winning bid and later retail price (used only to test the buy rule) |
| `analysis.py` | Split, regression fit, test-set evaluation, buy/pass rules, charts |
| `output/model_full.txt`, `output/model_reduced.txt` | Full regression summaries (coefficients, p-values, R²) |
| `output/test_fit_comparison.csv` | Baseline vs v1 on the 72 held-out test cars |
| `output/auction_decision_comparison.csv` | Baseline vs v1 buy rule on the 40 auction lots |
| `output/auction_decisions_by_lot.csv` | Each lot's prediction, decision under each rule, and profit |
| `output/pred_vs_actual_test.png` | Predicted vs actual sale price, test cars |
| `output/auction_profit_comparison.png` | Total auction profit by rule |

## Method

- **Split:** 240 cars → 168 training / 72 test (seed 42).
- **Model:** ordinary least squares regression of `Sale_Price` on mileage, age, horsepower, accident history, and number of owners.
- **v1:** `Num_Owners` was not significant (p ≈ 0.23), and dropping it left adjusted R² unchanged (0.880 → 0.880), so v1 uses mileage, age, horsepower, and accident history.
- **Baseline:** predicts the training-set average ($26,142) for every car.
- **Buy rule:** buy if auction price + $1,500 prep cost < predicted sale price. Profit per car bought = later sale price − auction price − $1,500.

## Results

**v1 coefficients (training fit, adjusted R² = 0.880):**

| Feature | Effect on sale price | p-value |
|---|---|---|
| Mileage | −$75 per 1,000 miles | <0.001 |
| Age | −$1,414 per year | <0.001 |
| Horsepower | +$73 per hp | <0.001 |
| Accident on record | −$2,298 | <0.001 |

**Fit on the 72 test cars:**

| | R² | RMSE |
|---|---|---|
| Baseline (average) | 0.00 | $5,874 |
| v1 regression | 0.77 | $2,810 |

**40 auction lots:**

| | Baseline rule | v1 rule |
|---|---|---|
| Cars bought | 15 | 17 |
| Money-losing cars bought | 11 | 6 |
| Total profit | −$9,500 | $15,800 |

For reference, buying only the 12 lots that turned out profitable would have made $21,500.
