# Data Quality and Model Governance

This note covers the checks run on the data, how the models should be monitored once in use, and the known limitations. It is a plan sized for a portfolio project, not a full MLOps stack.

## 1. Data lineage

| Stage | Location | Produced by |
|---|---|---|
| Raw source files (Kaggle, unmodified) | `data/raw/lending-club`, `data/raw/ieee-cis` | Manual download |
| Cleaned and feature-engineered Parquet | `data/processed/credit`, `data/processed/fraud` | `scripts/run_spark_processing.py` |
| Staging tables | Oracle `stg_loan`, `stg_transaction` | `scripts/load_warehouse.py` |
| Star schema (dimensions + facts) | Oracle `dim_*`, `fact_loan`, `fact_transaction` | `sql/03_load_warehouse.sql` |
| Models and evaluation reports | `data/models`, `data/reports` | `scripts/train_models.py` |

Each stage is rebuilt from the previous one by a script in the repository; nothing is edited by hand. The pipeline refuses to run on missing or empty source files and never generates synthetic replacement data.

## 2. Data quality checks

### At ingestion (Spark)

- **Schema contract:** required columns must exist (`loan_status`, `issue_d`, `fico_range_*`, `TransactionID`, `TransactionDT`, `isFraud`, ...). Otherwise the job fails.
- **Label definition:** only loans with a final outcome are kept. Open loans (`Current`, `Late`, `In Grace Period`) are excluded rather than labeled non-default.
- **Leakage control:** only fields known at loan origination are carried forward. Payment, recovery, hardship and settlement fields are dropped.
- **Type safety:** malformed values become null instead of failing silently or being coerced to a wrong value (`try_cast`, `try_to_timestamp`).
- **Domain cleaning:** Lending Club's negative DTI values (a missing-value code) become null; non-positive income gives null ratios instead of division errors.

### After loading the warehouse (`sql/04_data_quality_checks.sql`)

| Check | Rule | Severity |
|---|---|---|
| Completeness | Staging row count = fact row count | FAIL |
| Uniqueness | No duplicate loan or transaction IDs | FAIL |
| Label rate (credit) | Default rate between 10% and 30% | WARN |
| Label rate (fraud) | Fraud rate between 1% and 10% | WARN |
| Domain | FICO within 300–850; transaction amount > 0 | FAIL |
| Null rates | Income and DTI nulls below 1% | WARN |
| Feature sanity | `tx_count_24h` ≥ 1 (counts the transaction itself) | FAIL |
| Referential | Rows mapped to `Unknown` dimension members are counted | WARN |

The loader exits with an error on any FAIL. The latest load passed all 12 checks: 1,348,099 loans (19.98% default rate) and 590,540 transactions (3.50% fraud rate).

## 3. Model monitoring plan

### What to track, and how often

| Metric | Credit model | Fraud model | Frequency |
|---|---|---|---|
| Score distribution drift (PSI vs training) | ✓ | ✓ | Monthly (credit), weekly (fraud) |
| Feature drift (PSI on top 10 SHAP features) | ✓ | ✓ | Monthly |
| Feature missingness vs training | ✓ | ✓ | Every load |
| Calibration: predicted vs observed rate by score decile | ✓ | ✓ | Quarterly (credit), monthly (fraud) |
| Ranking power: ROC-AUC / PR-AUC on newly labeled data | ✓ | ✓ | Quarterly (credit), monthly (fraud) |
| Alert volume and precision on reviewed alerts | | ✓ | Weekly |

### PSI thresholds (`credit_risk_fraud.monitoring`)

| PSI | Reading | Action |
|---|---|---|
| < 0.10 | Stable | None |
| 0.10–0.25 | Moderate shift | Investigate the cause: product mix, policy change, data issue |
| ≥ 0.25 | Significant shift | Review the model; consider recalibration or retraining |

Baseline: train-vs-test score PSI is 0.003 for credit and 0.004 for fraud. Both are stable.

### Label delay

The two models get feedback at very different speeds, which shapes how they can be monitored:

- **Credit:** a 36-month loan takes up to three years to resolve. Ranking power and calibration can only be measured on old vintages. Until outcomes arrive, drift metrics (PSI) and early-delinquency proxies (for example, 30+ days past due at 6 months on book) are the main warning signals.
- **Fraud:** chargebacks and analyst reviews label transactions within weeks, so precision and recall can be tracked almost directly.

### Retraining and threshold triggers

Retrain or recalibrate when any of these holds:

1. Score PSI ≥ 0.25, or PSI ≥ 0.10 for three consecutive periods.
2. The calibration gap (|predicted − observed|) in any decile exceeds 5 percentage points on newly labeled data.
3. Fraud PR-AUC on the latest labeled month falls more than 20% below the test baseline (0.53).
4. A business change: a new product, a policy change, or a new region or channel.

Thresholds are reviewed together with the business owner whenever alert volume or analyst capacity changes. The fraud threshold is set for a target recall (currently 50%), so it is a capacity decision as much as a statistical one.

## 4. Fairness and explainability

- **Excluded from the credit model:** `addr_state`, because geography can act as a proxy for protected characteristics in lending. It stays in the warehouse for portfolio analysis only.
- **Also excluded:** Lending Club's own `grade`, `sub_grade` and `int_rate`, because they are outputs of another risk model.
- **Explainability:** SHAP importance and standardized logistic coefficients are published with every training run. Individual credit decisions can be explained with per-row SHAP values (reason codes).
- **Data sensitivity:** the source data is public and anonymized; it contains no names or account numbers. In a production setting, warehouse access would be role-based and personal data masked in non-production environments.

## 5. Known limitations

- **Right-censoring:** 60-month loans issued in 2014–2015 are not all resolved by the dataset cut-off (2018 Q4), so their default rate is understated.
- **Approximate fraud dates:** IEEE-CIS `TransactionDT` is an offset from an undisclosed start. Calendar dates assume a 2017-12-01 start, so day-of-week and month attributes for fraud are approximate.
- **Anonymized fraud features:** top fraud features (`C13`, `C1`, `D1`) are anonymized IEEE-CIS columns. Their business meaning is unknown, which limits explanation to analysts.
- **No HDFS yet:** processed data is stored as local Parquet; see the README.
- **Single snapshot:** monitoring is designed but not scheduled. A production setup would run the checks and PSI on a scheduler (for example cron or Airflow) and alert on breaches.
