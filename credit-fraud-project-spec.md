# Credit Scoring & Fraud Detection — On-Premise Data Pipeline

## Goal

Build a portfolio-grade data engineering + data science project demonstrating
skills targeted at on-premise banking/financial data roles (e.g. ACLEDA Bank
AI/ML posting: credit scoring, fraud detection, customer behavior analytics,
on-premise big data infrastructure, Oracle, SQL, statistical analysis, data
visualization).

**Target narrative:** "Built an on-premise-style data pipeline and modeling
system covering credit scoring and fraud detection, using Spark/HDFS for
processing and storage and Oracle as the analytical warehouse — mirroring the
architecture pattern used by banks with on-premise infrastructure
requirements."

This is a **separate project** from FinDocQA (the RAG/agentic AI/MLOps
project). Keep them distinct:
- This project = classical ML/statistics + data engineering + on-prem
  architecture (no LLM, no RAG, no agentic AI here)
- FinDocQA = LLM/RAG/agentic AI/MLOps

## Datasets (real data only — do NOT synthetically augment for now)

1. **Lending Club accepted loans** (2007–2020 full historical accepted loans
   dataset, ~2.5M rows, ~150 columns) — used for **credit scoring** (loan
   default prediction)
2. **IEEE-CIS Fraud Detection** (Kaggle competition dataset: transaction +
   identity tables, ~590K rows in train, hundreds of features) — used for
   **fraud detection**

Combine/use these two real datasets as-is. Do not generate synthetic
transaction volume at this stage — that's a possible future extension, not
part of the current scope.

**Honest framing to carry through the whole project (state this explicitly
in the README):** these datasets (combined raw size roughly ~4 GB) do not
strictly require distributed processing at this scale — pandas could handle
them on a single machine. Spark + HDFS are used deliberately here to
demonstrate the on-premise big-data architecture pattern that banking roles
like this require, not because this specific data volume demands it. Be
upfront about this rather than implying the tooling was a technical
necessity at this size.

## Non-goals / scope boundaries

- No LLM, RAG, or agentic AI components — that lives in the separate
  FinDocQA project.
- No cloud deployment (no AWS/S3/etc.) — this project should be framed and
  built as on-premise, consistent with the target job posting. Local
  Spark (standalone or local mode) + local HDFS (single-node or
  Docker-simulated multi-node) + local/VM Oracle instance.
- No synthetic data augmentation for now — real data only, as listed above.
- Avoid overclaiming: don't present this as "big data" in the sense of
  requiring a real cluster — be explicit that it's an architecture
  demonstration.

## Tech stack

- **Processing:** PySpark — cleaning, joins, feature engineering (including
  fraud-relevant aggregation/velocity features: transaction counts and
  amounts over rolling time windows per account/card)
- **Storage:** HDFS (local single-node, or simulated multi-node via Docker)
  for raw and intermediate processed data; Parquet format for
  processed/feature-engineered outputs
- **Warehouse:** Oracle — structured schema (customers/accounts,
  loans/transactions, fraud labels, engineered features) with a proper
  relational/star-schema design; complex SQL (window functions, joins,
  aggregations) written explicitly and shown, not hidden behind an ORM
- **Modeling:**
  - Credit scoring: logistic regression and/or gradient boosting (e.g.
    XGBoost/LightGBM) for loan default prediction; handle class imbalance
    properly; include interpretability (SHAP values or coefficient
    analysis) since credit decisions require explainability
  - Fraud detection: classification or anomaly detection on the imbalanced
    IEEE-CIS fraud dataset; evaluate with precision/recall/PR-AUC — NOT
    plain accuracy, since fraud data is heavily imbalanced and accuracy is
    misleading here
- **Statistical analysis:** hypothesis testing on default/fraud drivers
  (e.g. does loan purpose, income bracket, or region significantly affect
  default rate), distribution analysis, possibly a basic population
  stability index or similar drift-monitoring statistic (ties to "data
  quality/governance" in the target job posting)
- **Visualization:** Power BI or Tableau dashboard — credit risk segments,
  default rate trends, fraud pattern/trend monitoring
- **Language/tools:** Python and R where relevant (job posting lists both),
  SQL throughout

## Step-by-step plan

### Phase 1 — Ingestion + Spark processing
- Load Lending Club and IEEE-CIS datasets
- Clean and standardize with PySpark (handle missing values, type
  corrections, encoding)
- Feature engineering in Spark: for fraud data, build time-windowed
  aggregation/velocity features (e.g. transaction count/amount in last N
  hours per account); for credit data, build standard credit features
  (debt-to-income ratios, credit history length buckets, etc.)
- Write processed outputs to HDFS in Parquet format

### Phase 2 — Oracle warehouse
- Design a relational schema (fact tables: loans, transactions; dimension
  tables: customers, time, region/loan purpose)
- Load processed/aggregated data from HDFS into Oracle
- Write representative analytical SQL queries (window functions,
  aggregations, joins) that would support the modeling and BI layers

### Phase 3 — Modeling
- Credit scoring model: train/evaluate default prediction model, document
  metrics (AUC, precision/recall, calibration) and interpretability output
- Fraud detection model: train/evaluate on imbalanced fraud data, document
  precision/recall/PR-AUC tradeoffs and reasoning for the chosen threshold
- Statistical tests on key drivers for both credit and fraud outcomes

### Phase 4 — Visualization + governance notes
- Build a BI dashboard (Power BI/Tableau) summarizing credit risk segments,
  default trends, and fraud monitoring metrics
- Write a short data quality/governance note: what checks were run on the
  data, and a simple plan for model monitoring/drift detection over time
  (doesn't need a full MLOps stack — a clear, honest plan is enough)

### Phase 5 — Documentation
- README covering: architecture diagram (Spark/HDFS → Oracle → models →
  BI), the honest "why Spark/HDFS at this scale" framing described above,
  model results, and reasoning behind key design choices (schema design,
  imbalance handling, threshold choices, etc.) — this is what needs to be
  defensible in an interview

## Known risk points

- Feature engineering for fraud velocity features (rolling time windows) is
  the most Spark-intensive and fiddly part — budget real time here
- Class imbalance handling for fraud detection needs care — naive
  approaches will produce misleadingly high accuracy with poor real recall
- Oracle schema design should be done deliberately before loading data, not
  improvised — a clean star schema is part of what's being demonstrated
