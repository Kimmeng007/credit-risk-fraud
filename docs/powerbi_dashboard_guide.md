# Power BI Dashboard: Build Guide

This guide builds a 5-page dashboard from the CSV extracts in `data/bi/`. It takes about an hour. Each step explains *why* the chart exists, so you can answer interview questions about it.

**Before you start**

1. Install **Power BI Desktop** (free) from the Microsoft Store.
2. Make sure the extracts are up to date. Docker and Oracle must be running:
   ```powershell
   python scripts/export_bi_extracts.py
   ```

| File | One row per | Used on |
|---|---|---|
| `credit_segments.csv` | issue month × grade × purpose × term × FICO band × DTI band | Pages 1, 2 |
| `credit_state.csv` | state × issue year | Page 2 |
| `fraud_daily.csv` | day × product × card type × device | Page 3 |
| `fraud_trend.csv` | day (with 7-day rolling totals) | Page 3 |
| `fraud_patterns.csv` | product × amount decile × velocity bucket | Page 4 |
| `model_metrics.csv` | task × model | Page 5 |
| `model_calibration.csv` | task × score decile | Page 5 |

---

## Step 1: Load the data

1. **Home → Get data → Text/CSV**, pick `data/bi/credit_segments.csv`, then click **Transform Data** (not Load).
2. In Power Query, check each column's type icon in the header:
   - `issue_month` → **Date**
   - `term_months` → **Text** (it's a category, 36 or 60, not a number to sum)
   - `loans`, `defaults` → **Whole number**
   - `loan_amount`, `defaulted_amount` → **Decimal number**
3. Click **Close & Apply**.
4. Repeat for the other 6 files. Set `transaction_date` to **Date** and the rate columns to **Decimal number**.

The tables don't need relationships to each other: each page uses one table, or tables that are filtered separately.

> **Why CSV and not a live Oracle connection?** Power BI's Oracle connector needs a separate Oracle client install. The CSVs are exported straight from the Oracle views (`sql/05_bi_views.sql`), so the numbers are identical. In a bank you would connect to the warehouse directly and schedule a refresh.

## Step 2: Create the measures

Select a table in the **Data** pane, then **Home → New measure**, and paste each line. Format the rate measures as **Percentage** with 2 decimals (Measure tools → Format).

**On `credit_segments`:**
```DAX
Loans = SUM(credit_segments[loans])
Defaults = SUM(credit_segments[defaults])
Default Rate = DIVIDE([Defaults], [Loans])
Loan Volume = SUM(credit_segments[loan_amount])
Defaulted Amount Share = DIVIDE(SUM(credit_segments[defaulted_amount]), [Loan Volume])
```

**On `credit_state`:**
```DAX
State Default Rate = DIVIDE(SUM(credit_state[defaults]), SUM(credit_state[loans]))
```

**On `fraud_daily`:**
```DAX
Transactions = SUM(fraud_daily[transactions])
Fraud Count = SUM(fraud_daily[fraud_count])
Fraud Rate = DIVIDE([Fraud Count], [Transactions])
Fraud Amount = SUM(fraud_daily[fraud_amount])
Fraud Amount Rate = DIVIDE([Fraud Amount], SUM(fraud_daily[transaction_amount]))
```

**On `fraud_trend`:**
```DAX
Daily Fraud Rate = DIVIDE(SUM(fraud_trend[fraud_count]), SUM(fraud_trend[transactions]))
Rolling 7d Fraud Rate = DIVIDE(SUM(fraud_trend[fraud_count_7d]), SUM(fraud_trend[transactions_7d]))
```

**On `fraud_patterns`:**
```DAX
Pattern Fraud Rate = DIVIDE(SUM(fraud_patterns[fraud_count]), SUM(fraud_patterns[transactions]))
```

> **Why sums divided by sums?** A rate must be computed as total defaults ÷ total loans. Averaging pre-computed rates would give a segment with 10 loans the same weight as one with 100,000. That's why the views export counts, not rates, and the measures divide at whatever level you're viewing.

---

## Page 1: Credit Risk Overview

**Purpose:** how big is the portfolio, how risky is it, and how has risk changed over time?

1. **Four cards** (Visualizations → Card): `Loans`, `Default Rate`, `Loan Volume`, `Defaulted Amount Share`.
2. **Line chart:** X-axis `issue_month`, Y-axis `Default Rate`.
   - Add a **filter on this visual**: `issue_year` ≤ 2015.
   - Add a text box under it: *"2016+ vintages excluded: most of those loans had not reached a final outcome by the dataset cut-off, so their default rate is biased."*
3. **Clustered bar chart:** Y-axis `grade`, X-axis `Default Rate`. Sort by grade A→G.
4. **Clustered bar chart:** Y-axis `purpose`, X-axis `Default Rate`. Sort descending.
5. **Slicers:** `term_months` and `issue_year`.

**Talking points:**
- Default rate rises steadily from 6% for grade A to 50% for grade G, so Lending Club's grading works. My model excludes grade so that it doesn't just copy their model.
- 60-month loans default twice as often as 36-month loans (32.5% vs 16.0%); term is the strongest driver in both the SHAP values and the statistical tests.

## Page 2: Credit Risk Segments

**Purpose:** which borrower segments carry the most risk?

1. **Matrix:** Rows `fico_band`, Columns `dti_band`, Values `Default Rate`.
   - Format → Cell elements → **Background color** on, so it becomes a heat map (green = low, red = high).
   - Add `Loans` as a second value, so viewers can see how many loans are in each cell.
   - Add a **visual filter** excluding `fico_band` = "1: <660". Lending Club's minimum FICO was about 660, so that band holds only a few hundred loans and its rates are unreliable.
2. **Filled map:** Location `credit_state[state_code]`, Color saturation `State Default Rate`.
   - If maps are disabled: File → Options → Security → enable "Use Map and Filled Map visuals".
3. **Text box:** *"State is shown for portfolio analysis only. It is not a model input, because geography can act as a proxy for protected characteristics in lending."*

**Talking points:**
- Risk rises as FICO falls and DTI rises: about 8% for FICO 740+ with DTI 10–19, versus 33% for FICO 660–699 with DTI 30+, a 4× difference.
- State differences are statistically significant but small (Cramér's V = 0.05). With a million rows, almost everything is significant, so effect size matters more.

## Page 3: Fraud Monitoring

**Purpose:** what an analyst checks each day. Is fraud rising, and where is it?

1. **Cards:** `Transactions`, `Fraud Count`, `Fraud Rate`, `Fraud Amount`.
2. **Line chart:** X-axis `fraud_trend[transaction_date]`, Y-axis `Daily Fraud Rate` and `Rolling 7d Fraud Rate`.
3. **Clustered column charts** of `Fraud Rate` by `product_code`, by `card_type` and by `device_type`.
4. **Slicer:** `fraud_daily[transaction_date]` (between).

**Talking points:**
- The daily rate is noisy; the 7-day rolling rate is the monitoring signal. It's computed as rolling fraud ÷ rolling transactions, not as an average of daily rates.
- The 7-day line comes from `fraud_trend`, which is not split by product, so the product slicers don't filter it.
- Dates assume the IEEE-CIS data starts on 2017-12-01, the common convention; the real start date is undisclosed.

## Page 4: Fraud Patterns

**Purpose:** do the engineered features actually separate fraud?

1. **Clustered column chart:** X-axis `velocity_bucket`, Y-axis `Pattern Fraud Rate`.
2. **Line chart:** X-axis `amount_decile`, Y-axis `Pattern Fraud Rate`.
3. **Matrix:** Rows `product_code`, Columns `velocity_bucket`, Values `Pattern Fraud Rate` with a background-color heat map.

**Talking points:**
- Fraud rate rises steadily with the number of transactions on the same card in 24h: 2.5% for a single transaction up to 6.8% for 11 or more. This velocity feature was built in Spark with time windows, and it's one of the strongest fraud drivers in the statistical tests.
- Amount has a **U-shape**: the smallest amounts (5.6%, often card-testing) and the largest (5.1%) are riskier than mid-range amounts (about 2–3%). The Mann-Whitney test found no effect (p = 0.23) because it only detects one-directional shifts, and the two ends cancel out. This is why the model uses trees, which capture non-linear patterns like this, and why a chart can show what a single test misses.

## Page 5: Model Performance

**Purpose:** how good are the models, and are their probabilities trustworthy?

1. **Table:** `model_metrics` with `task`, `model`, `roc_auc`, `pr_auc`, `brier`, `precision`, `recall`, `score_psi`. Format the numbers to 3 decimals.
2. **Clustered column chart:** X-axis `model`, Y-axis `pr_auc`, Legend `task`.
3. **Line chart:** X-axis `model_calibration[bin]`, Y-axis `mean_predicted` and `observed_rate`, plus a **slicer** on `model_calibration[task]`.
4. **Text box** with the key results: *"Fraud PR-AUC 0.17 → 0.53 (LightGBM vs logistic regression, out-of-time test). Credit probabilities calibrated: top decile predicts 50.5%, observed 50.7%."*

**Talking points:**
- Fraud is measured with PR-AUC, not accuracy: flagging nothing would already score 96.5% accuracy.
- When the two calibration lines overlap, a predicted 20% default risk really means about 20%. That matters for pricing and expected loss, not only approve/decline decisions.
- Score PSI is about 0.003, so train and test score distributions are stable.

---

## Finishing touches

- Use one consistent color for "bad" (defaults/fraud), e.g. red, and one neutral color for volume.
- Give every page a title that states the conclusion, not just the topic. For example *"60-month loans default twice as often"* rather than *"Default by term"*.
- **Save** as `dashboard/credit_fraud_dashboard.pbix` in the project.
- **Export screenshots** of each page to `docs/dashboard/` and add them to the README, so people can see the dashboard without installing Power BI.
