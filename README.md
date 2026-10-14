# Credit Risk and Fraud Detection

An on-premise-style data engineering and classical machine learning project for banking analytics. The pipeline runs **PySpark -> Parquet -> Oracle -> models -> BI** and covers two real datasets:

- Lending Club accepted loans for default prediction
- IEEE-CIS Fraud Detection transaction and identity data for fraud prediction

## Honest scope

The combined raw data is roughly 4 GB and does not strictly require distributed processing at this scale. pandas could process it on one machine. Spark is used deliberately to demonstrate an on-premise banking architecture pattern, not because this dataset size demands a cluster. No synthetic rows are generated, and no cloud, LLM, RAG, or agentic components are included.

**HDFS is not used yet.** Processed Parquet lives on the local disk (`data/processed`) and stands in for an HDFS data lake. Spark reads and writes through Hadoop's file system API, so moving to a single-node HDFS changes only the paths (`hdfs://...`), not the processing code. HDFS is planned as a later step. On one machine it would add no space, speed or fault tolerance; it would only demonstrate the architecture.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,spark]"
```

Place the original downloaded files under `data/raw/lending-club` and `data/raw/ieee-cis`. The pipeline rejects missing or empty inputs and never creates synthetic replacement data.

```powershell
python scripts/run_spark_processing.py --credit --fraud
python -m pytest
```

Processed Parquet outputs are written to `data/processed`. Spark runs in local mode with a 6 GB driver heap by default; set `SPARK_DRIVER_MEMORY` to change it.

## Modeling

Each task trains a class-weighted logistic regression baseline and a LightGBM model. The LightGBM scores are then calibrated with isotonic regression fitted on the validation split. Every report includes ROC-AUC, PR-AUC, Brier score, base rate, precision, recall, the selected threshold, a test-set calibration table, the score PSI between train and test, and SHAP feature importance.

```powershell
python scripts/train_models.py --credit --fraud     # reports in data/reports, models in data/models
python scripts/run_statistical_tests.py             # hypothesis tests on default and fraud drivers
```

### Results (out-of-time test split)

| Model | ROC-AUC | PR-AUC | Brier | Precision @ recall | Base rate |
|---|---|---|---|---|---|
| Credit: logistic regression | 0.721 | 0.401 | 0.209 | 0.39 @ 0.47 | 20.0% |
| Credit: LightGBM, calibrated | 0.732 | 0.409 | 0.141 | 0.40 @ 0.50 | 20.0% |
| Fraud: logistic regression | 0.823 | 0.167 | 0.151 | 0.20 @ 0.50 | 3.5% |
| Fraud: LightGBM, calibrated | 0.903 | 0.528 | 0.022 | 0.59 @ 0.48 | 3.5% |

How to read these results:

- **Credit gains from LightGBM are small.** Public Lending Club data without Lending Club's own grade typically tops out around 0.70–0.73 ROC-AUC. The larger improvement is calibration: the class-weighted logistic model inflates probabilities (Brier 0.209), while calibrated LightGBM matches observed default rates in every test decile (top decile: 50.5% predicted vs 50.7% observed). This matters because credit scores feed pricing and expected-loss calculations, not only accept/reject decisions.
- **Fraud gains are large.** PR-AUC rises from 0.17 to 0.53, a 15× lift over the 3.5% base rate. At the chosen threshold, about 6 of every 10 flagged transactions are fraud and about half of all fraud is caught. Accuracy is not reported: a model that flags nothing scores 96.5% accuracy.
- **The threshold is a business choice.** It is set as the highest-precision point that catches at least 50% of positives on the validation split (`--minimum-recall`). Test recall comes in slightly below the target (0.48 vs 0.50) because the test period differs from validation. Raising the recall floor catches more fraud at the cost of more false alerts for analysts to review.
- **Fraud scores carry their own drift risk.** The top fraud SHAP features are anonymized IEEE-CIS columns (`C13`, `C1`, `D1`) and card and email identifiers. These are strong in-sample but could shift as card portfolios change. Score PSI between train and test is 0.004 (stable) for both models; see the monitoring plan below.
- **SHAP drivers agree with the statistical tests.** For credit, the strongest drivers are term, installment-to-income, FICO and loan-to-income. Bar charts are saved as `data/reports/*_shap_importance.png`.

### Statistical tests

`scripts/run_statistical_tests.py` runs 14 tests. Categorical drivers use chi-square tests with Cramér's V; numeric drivers use Mann-Whitney U with rank-biserial r. Significance uses a Bonferroni-adjusted alpha. With about a million rows almost every test is significant, so the effect size is what shows whether a driver matters:

| Outcome | Driver | Effect size | Magnitude |
|---|---|---|---|
| Default | Loan term (36 vs 60 months) | V = 0.20 | small, the strongest credit driver |
| Default | FICO score | r = −0.18 | small |
| Default | Debt-to-income | r = 0.16 | small |
| Default | Purpose, state, income quintile, home ownership, credit history | V = 0.03–0.07 | negligible, though all significant |
| Fraud | Seconds since previous transaction (velocity) | r = −0.26 | small, the strongest fraud driver |
| Fraud | Transactions in last 24h (velocity) | r = 0.21 | small |
| Fraud | Product code | V = 0.17 | small |
| Fraud | Transaction amount | r = −0.005 | negligible and not significant (p = 0.23) |

The engineered velocity features are the strongest single fraud drivers in these tests, which supports the Spark feature engineering. State is significant but has a negligible effect, and it is excluded from the credit model because geography can act as a proxy for protected characteristics in lending.

### Key design decisions

- **Credit label uses resolved loans only.** `Fully Paid` (and the "does not meet credit policy" variant) is 0; `Charged Off`, `Default` and the policy variant are 1. `Current`, `Late` and `In Grace Period` loans have no final outcome and are dropped rather than labeled non-default, which would understate the default rate.
- **Origination-time fields only.** The processed credit table keeps fields known at application time. Payment, recovery, hardship and settlement columns are dropped because they are recorded after issuance and leak the outcome.
- **Lending Club's grade, sub-grade and interest rate are excluded from model features.** They are outputs of Lending Club's own risk model; they stay in the warehouse for BI segmentation. `addr_state` is also excluded from the credit model (fair-lending proxy risk) but is analyzed statistically.
- **No class reweighting for LightGBM.** Reweighting distorts predicted probabilities, and ranking metrics don't need it. Imbalance is handled by early stopping on PR-AUC and by choosing the threshold for a target recall. The logistic baseline keeps balanced class weights, which is why its Brier score is worse.
- **Early stopping, calibration and threshold all use the validation split.** The test split is used only once, for the final report.
- **Mature vintages only.** Credit training uses loans issued before 2016 (`--credit-max-issue-date`). Later vintages are mostly unresolved, and their resolved subset over-represents early payoffs and fast defaults. 60-month loans from 2014–2015 are still partly right-censored; this is a known limitation.
- **Out-of-time evaluation.** Both models split train/validation/test chronologically (`issue_date` for credit, `TransactionDT` for fraud) so evaluation mirrors scoring future applications and transactions.
- **Threshold chosen on validation, reported on test.** The threshold is the highest-precision point that meets the recall floor on the validation split; test metrics are never used to tune it.
- **Fraud velocity features are keyed by `card1 | addr1 | P_emaildomain`.** `card1` alone is shared by many cardholders. Counts and amounts are computed over backward-looking 1h, 24h and 7d windows, plus amount relative to the card's 7-day mean and time since the previous transaction.

## Oracle warehouse

Oracle Database 23ai Free runs locally in Docker. No cloud services are used.

```powershell
pip install -e ".[oracle]"
docker compose up -d oracle          # first start downloads the image and takes a few minutes
docker compose ps                    # wait until the status shows "healthy"
python scripts/load_warehouse.py     # rebuilds schema, loads Parquet, runs data quality checks
```

The loader uses python-oracledb in thin mode, so no Oracle client install is needed. Connection settings come from `ORACLE_DSN` (default `localhost:1521/FREEPDB1`), `ORACLE_USER` (default `CREDIT_DW`) and `APP_USER_PASSWORD`. The defaults in `docker-compose.yml` are for local development only.

| File | Purpose |
|---|---|
| `sql/01_schema.sql` | Staging tables plus two star schemas sharing a conformed `dim_date` |
| `sql/03_load_warehouse.sql` | Set-based ELT from staging into dimensions and facts |
| `sql/04_data_quality_checks.sql` | Post-load checks: completeness, uniqueness, label rates, domains, null rates |
| `sql/02_analytical_queries.sql` | Vintage default rates, state risk ranking, YoY change, FICO x DTI matrix, rolling 7-day fraud rate, amount deciles, velocity vs fraud, highest-risk cards |

**Schema design**

- Credit star: `fact_loan` (one row per resolved loan) with `dim_date`, `dim_loan_purpose`, `dim_region`, `dim_credit_grade` and `dim_borrower_profile`.
- Fraud star: `fact_transaction` (one row per transaction) with `dim_date`, `dim_card` and `dim_product`.
- Public Lending Club data has no persistent customer ID, so `dim_borrower_profile` is a junk dimension. It has one row per combination of home ownership, verification status, application type, employment-length band and credit-history bucket.
- Missing categorical values map to an explicit `Unknown` member, so every fact row joins to every dimension.
- `dim_date` uses a `YYYYMMDD` smart key. IEEE-CIS `TransactionDT` is a seconds offset from an undisclosed start time. It is mapped onto calendar dates assuming a 2017-12-01 start, the common convention. Calendar attributes for fraud are therefore approximate; ordering and velocity features are exact.
- Data lands in staging first, and SQL inside Oracle builds the dimensions and facts. This keeps transformation logic visible and reviewable as SQL, which is the usual pattern in bank data warehouses.

**Current load:** 1,348,099 resolved loans (19.98% default rate) and 590,540 transactions (3.50% fraud rate). All 12 data quality checks pass, and no rows are dropped between staging and the fact tables.

## Governance

The full data quality and model monitoring plan is in [docs/governance.md](docs/governance.md). `scripts/load_warehouse.py` exits with an error if any data quality check returns FAIL, so a bad load is not used downstream. WARN results need a human review. Monitor monthly default and fraud rates, feature missingness, and population stability by score band. Investigate drift before changing thresholds or retraining.

## Build phases

1. Run Spark ingestion and feature engineering.
2. Load Parquet into Oracle with `scripts/load_warehouse.py`.
3. Train models and publish metrics and interpretability outputs.
4. Create the BI views and export extracts with `python scripts/export_bi_extracts.py`, then build the Power BI dashboard following [docs/powerbi_dashboard_guide.md](docs/powerbi_dashboard_guide.md).