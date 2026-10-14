-- Reporting views for the Power BI dashboard. Each view is pre-aggregated to the
-- grain one dashboard page needs, so the BI layer holds counts and sums and computes
-- rates as measures (sum of defaults / sum of loans), never averages of rates.

-- Credit: portfolio by issue month and risk segment.
CREATE OR REPLACE VIEW v_bi_credit_segments AS
SELECT
    d.month_start_date AS issue_month,
    d.year_number AS issue_year,
    g.grade,
    p.purpose_name AS purpose,
    l.term_months,
    CASE WHEN l.fico_mid < 660 THEN '1: <660'
         WHEN l.fico_mid < 700 THEN '2: 660-699'
         WHEN l.fico_mid < 740 THEN '3: 700-739'
         ELSE '4: 740+' END AS fico_band,
    CASE WHEN l.debt_to_income IS NULL THEN '0: Unknown'
         WHEN l.debt_to_income < 10 THEN '1: <10'
         WHEN l.debt_to_income < 20 THEN '2: 10-19'
         WHEN l.debt_to_income < 30 THEN '3: 20-29'
         ELSE '4: 30+' END AS dti_band,
    COUNT(*) AS loans,
    SUM(l.default_flag) AS defaults,
    SUM(l.loan_amount) AS loan_amount,
    SUM(CASE WHEN l.default_flag = 1 THEN l.loan_amount ELSE 0 END) AS defaulted_amount
FROM fact_loan l
JOIN dim_date d ON d.date_key = l.issue_date_key
JOIN dim_credit_grade g ON g.grade_key = l.grade_key
JOIN dim_loan_purpose p ON p.purpose_key = l.purpose_key
GROUP BY
    d.month_start_date, d.year_number, g.grade, p.purpose_name, l.term_months,
    CASE WHEN l.fico_mid < 660 THEN '1: <660'
         WHEN l.fico_mid < 700 THEN '2: 660-699'
         WHEN l.fico_mid < 740 THEN '3: 700-739'
         ELSE '4: 740+' END,
    CASE WHEN l.debt_to_income IS NULL THEN '0: Unknown'
         WHEN l.debt_to_income < 10 THEN '1: <10'
         WHEN l.debt_to_income < 20 THEN '2: 10-19'
         WHEN l.debt_to_income < 30 THEN '3: 20-29'
         ELSE '4: 30+' END;

-- Credit: geography for the map visual (analysis only; state is not a model feature).
CREATE OR REPLACE VIEW v_bi_credit_state AS
SELECT
    r.state_code,
    d.year_number AS issue_year,
    COUNT(*) AS loans,
    SUM(l.default_flag) AS defaults,
    SUM(l.loan_amount) AS loan_amount
FROM fact_loan l
JOIN dim_region r ON r.region_key = l.region_key
JOIN dim_date d ON d.date_key = l.issue_date_key
GROUP BY r.state_code, d.year_number;

-- Fraud: daily monitoring by product, card type and device.
CREATE OR REPLACE VIEW v_bi_fraud_daily AS
SELECT
    d.calendar_date AS transaction_date,
    d.day_of_week,
    d.is_weekend,
    p.product_code,
    c.card_type,
    t.device_type,
    COUNT(*) AS transactions,
    SUM(t.fraud_flag) AS fraud_count,
    SUM(t.transaction_amount) AS transaction_amount,
    SUM(CASE WHEN t.fraud_flag = 1 THEN t.transaction_amount ELSE 0 END) AS fraud_amount
FROM fact_transaction t
JOIN dim_date d ON d.date_key = t.transaction_date_key
JOIN dim_product p ON p.product_key = t.product_key
JOIN dim_card c ON c.card_key = t.card_key
GROUP BY d.calendar_date, d.day_of_week, d.is_weekend, p.product_code, c.card_type, t.device_type;

-- Fraud: totals per day with a 7-day rolling fraud rate (ratio of rolling sums).
CREATE OR REPLACE VIEW v_bi_fraud_trend AS
SELECT
    transaction_date,
    transactions,
    fraud_count,
    fraud_amount,
    SUM(fraud_count) OVER w AS fraud_count_7d,
    SUM(transactions) OVER w AS transactions_7d,
    ROUND(SUM(fraud_count) OVER w / SUM(transactions) OVER w, 5) AS fraud_rate_7d
FROM (
    SELECT d.calendar_date AS transaction_date,
           COUNT(*) AS transactions,
           SUM(t.fraud_flag) AS fraud_count,
           SUM(CASE WHEN t.fraud_flag = 1 THEN t.transaction_amount ELSE 0 END) AS fraud_amount
    FROM fact_transaction t
    JOIN dim_date d ON d.date_key = t.transaction_date_key
    GROUP BY d.calendar_date
)
WINDOW w AS (ORDER BY transaction_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW);

-- Fraud: risk patterns by amount decile and 24h velocity bucket.
CREATE OR REPLACE VIEW v_bi_fraud_patterns AS
SELECT
    product_code,
    amount_decile,
    velocity_bucket,
    COUNT(*) AS transactions,
    SUM(fraud_flag) AS fraud_count,
    SUM(transaction_amount) AS transaction_amount
FROM (
    SELECT
        p.product_code,
        t.fraud_flag,
        t.transaction_amount,
        NTILE(10) OVER (ORDER BY t.transaction_amount) AS amount_decile,
        CASE WHEN t.tx_count_24h = 1 THEN '1: single'
             WHEN t.tx_count_24h <= 3 THEN '2: 2-3'
             WHEN t.tx_count_24h <= 10 THEN '3: 4-10'
             ELSE '4: 11+' END AS velocity_bucket
    FROM fact_transaction t
    JOIN dim_product p ON p.product_key = t.product_key
)
GROUP BY product_code, amount_decile, velocity_bucket;
