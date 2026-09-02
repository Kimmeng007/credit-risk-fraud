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

`credit_risk_fraud.modeling` provides an imbalance-aware logistic regression baseline and threshold selection based on a minimum recall target. Reports must include ROC-AUC, PR-AUC, precision, recall, and the selected threshold. Credit model interpretation should publish standardized coefficients; fraud decisions should prioritize precision/recall and PR-AUC over accuracy.

## Warehouse and governance

The Oracle star-schema DDL and representative window-function queries live in `sql/`. Before loading, validate uniqueness, null rates, accepted domain values, date ranges, and the default/fraud label distribution. Monitor monthly default and fraud rates, feature missingness, and population stability by score band; investigate drift before changing thresholds or retraining.

## Build phases

1. Run Spark ingestion and feature engineering.
2. Load Parquet into the Oracle schema in `sql/01_schema.sql`.
3. Train models and publish metrics and interpretability outputs.
4. Connect Power BI or Tableau to the Oracle analytical queries for risk segments, default trends, and fraud monitoring.