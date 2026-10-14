# Credit Risk and Fraud Detection

An on-premise-style data engineering and classical machine learning project for banking analytics. The pipeline is designed around **PySpark -> HDFS/Parquet -> Oracle -> models -> BI** and covers two real datasets:

- Lending Club accepted loans for default prediction
- IEEE-CIS Fraud Detection transaction and identity data for fraud prediction

## Honest scope

The combined raw data is roughly 4 GB and does not strictly require distributed processing at this scale. pandas could process it on one machine. Spark and HDFS are used deliberately to demonstrate an on-premise banking architecture pattern, not because this dataset size demands a cluster. No synthetic rows are generated, and no cloud, LLM, RAG, or agentic components are included.

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

Processed Parquet outputs are written to `data/processed`. Configure HDFS paths through the Spark submit environment when moving from local mode to a single-node HDFS deployment.

## Modeling contract

`credit_risk_fraud.modeling` provides an imbalance-aware logistic regression baseline and threshold selection based on a minimum recall target. Reports must include ROC-AUC, PR-AUC, Brier score, base rate, precision, recall, and the selected threshold. Credit model interpretation should publish standardized coefficients; fraud decisions should prioritize precision/recall and PR-AUC over accuracy.

```powershell
python scripts/train_models.py --credit --fraud
```

### Key design decisions

- **Credit label uses resolved loans only.** `Fully Paid` (and the "does not meet credit policy" variant) is 0; `Charged Off`, `Default` and the policy variant are 1. `Current`, `Late` and `In Grace Period` loans have no final outcome and are dropped rather than labeled non-default, which would understate the default rate.
- **Origination-time fields only.** The processed credit table keeps fields known at application time. Payment, recovery, hardship and settlement columns are dropped because they are recorded after issuance and leak the outcome.
- **Lending Club's grade, sub-grade and interest rate are excluded from model features.** They are outputs of Lending Club's own risk model; they stay in the warehouse for BI segmentation.
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

`scripts/load_warehouse.py` exits with an error if any data quality check returns FAIL, so a bad load is not used downstream. WARN results need a human review. Monitor monthly default and fraud rates, feature missingness, and population stability by score band. Investigate drift before changing thresholds or retraining.

## Build phases

1. Run Spark ingestion and feature engineering.
2. Load Parquet into Oracle with `scripts/load_warehouse.py`.
3. Train models and publish metrics and interpretability outputs.
4. Connect Power BI or Tableau to the Oracle analytical queries for risk segments, default trends, and fraud monitoring.