-- Post-load data quality checks. Each query returns (check_name, observed_value, status).
-- FAIL means the load must not be used downstream; WARN needs a human look.

-- Completeness: every staged row reached its fact table.
SELECT 'fact_loan rows dropped vs staging',
       (SELECT COUNT(*) FROM stg_loan) - (SELECT COUNT(*) FROM fact_loan),
       CASE WHEN (SELECT COUNT(*) FROM stg_loan) = (SELECT COUNT(*) FROM fact_loan)
            THEN 'PASS' ELSE 'FAIL' END
FROM dual;

SELECT 'fact_transaction rows dropped vs staging',
       (SELECT COUNT(*) FROM stg_transaction) - (SELECT COUNT(*) FROM fact_transaction),
       CASE WHEN (SELECT COUNT(*) FROM stg_transaction) = (SELECT COUNT(*) FROM fact_transaction)
            THEN 'PASS' ELSE 'FAIL' END
FROM dual;

-- Uniqueness of source identifiers in staging (the fact primary keys would also reject these).
SELECT 'duplicate loan ids in staging',
       COUNT(*) - COUNT(DISTINCT loan_id),
       CASE WHEN COUNT(*) = COUNT(DISTINCT loan_id) THEN 'PASS' ELSE 'FAIL' END
FROM stg_loan;

SELECT 'duplicate transaction ids in staging',
       COUNT(*) - COUNT(DISTINCT transaction_id),
       CASE WHEN COUNT(*) = COUNT(DISTINCT transaction_id) THEN 'PASS' ELSE 'FAIL' END
FROM stg_transaction;

-- Label distribution: resolved Lending Club loans default at roughly 15-25%;
-- IEEE-CIS training data is about 3.5% fraud. Large deviations signal a labeling bug.
SELECT 'loan default rate',
       ROUND(AVG(default_flag), 4),
       CASE WHEN AVG(default_flag) BETWEEN 0.10 AND 0.30 THEN 'PASS' ELSE 'WARN' END
FROM fact_loan;

SELECT 'transaction fraud rate',
       ROUND(AVG(fraud_flag), 4),
       CASE WHEN AVG(fraud_flag) BETWEEN 0.01 AND 0.10 THEN 'PASS' ELSE 'WARN' END
FROM fact_transaction;

-- Domain checks on key numeric fields.
SELECT 'loans with FICO outside 300-850',
       COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM fact_loan
WHERE fico_mid NOT BETWEEN 300 AND 850;

SELECT 'transactions with non-positive amount',
       COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM fact_transaction
WHERE transaction_amount <= 0;

-- Null rates on model-critical fields (share of rows).
SELECT 'loan annual_income null rate',
       ROUND(AVG(CASE WHEN annual_income IS NULL THEN 1 ELSE 0 END), 4),
       CASE WHEN AVG(CASE WHEN annual_income IS NULL THEN 1 ELSE 0 END) < 0.01 THEN 'PASS' ELSE 'WARN' END
FROM fact_loan;

SELECT 'loan debt_to_income null rate',
       ROUND(AVG(CASE WHEN debt_to_income IS NULL THEN 1 ELSE 0 END), 4),
       CASE WHEN AVG(CASE WHEN debt_to_income IS NULL THEN 1 ELSE 0 END) < 0.01 THEN 'PASS' ELSE 'WARN' END
FROM fact_loan;

-- Velocity features must count at least the current transaction.
SELECT 'transactions with tx_count_24h < 1',
       COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END
FROM fact_transaction
WHERE tx_count_24h < 1 OR tx_count_24h IS NULL;

-- Unknown dimension members: tolerated, but tracked.
SELECT 'loans mapped to Unknown purpose/region/grade',
       COUNT(*),
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END
FROM fact_loan l
JOIN dim_loan_purpose p ON p.purpose_key = l.purpose_key
JOIN dim_region r ON r.region_key = l.region_key
JOIN dim_credit_grade g ON g.grade_key = l.grade_key
WHERE p.purpose_name = 'Unknown' OR r.state_code = 'Unknown' OR g.sub_grade = 'Unknown';
