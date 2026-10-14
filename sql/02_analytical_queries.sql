-- Representative analytical queries over the warehouse. Each feeds a modeling
-- question or a BI view; all logic is explicit SQL (joins, windows, aggregation).

-- 1. Vintage default rate by issue quarter and grade.
--    Why: credit performance is compared by origination cohort, not calendar time.
SELECT
    d.year_number,
    d.quarter_number,
    g.grade,
    COUNT(*) AS loans,
    ROUND(AVG(l.default_flag), 4) AS default_rate,
    ROUND(SUM(CASE WHEN l.default_flag = 1 THEN l.loan_amount ELSE 0 END)
          / SUM(l.loan_amount), 4) AS defaulted_amount_share
FROM fact_loan l
JOIN dim_date d ON d.date_key = l.issue_date_key
JOIN dim_credit_grade g ON g.grade_key = l.grade_key
GROUP BY d.year_number, d.quarter_number, g.grade
ORDER BY d.year_number, d.quarter_number, g.grade;

-- 2. States ranked by default rate, with lift over the portfolio average.
--    AVG() OVER () computes the portfolio rate without a second pass.
SELECT state_code, loans, default_rate, portfolio_rate,
       ROUND(default_rate / portfolio_rate, 3) AS lift,
       RANK() OVER (ORDER BY default_rate DESC) AS risk_rank
FROM (
    SELECT
        r.state_code,
        COUNT(*) AS loans,
        ROUND(AVG(l.default_flag), 4) AS default_rate,
        ROUND(SUM(SUM(l.default_flag)) OVER () / SUM(COUNT(*)) OVER (), 4) AS portfolio_rate
    FROM fact_loan l
    JOIN dim_region r ON r.region_key = l.region_key
    GROUP BY r.state_code
    HAVING COUNT(*) >= 1000
)
ORDER BY risk_rank;

-- 3. Year-over-year change in default rate per loan purpose (LAG).
SELECT
    purpose_name,
    year_number,
    loans,
    default_rate,
    ROUND(default_rate - LAG(default_rate) OVER (
        PARTITION BY purpose_name ORDER BY year_number), 4) AS yoy_change
FROM (
    SELECT p.purpose_name, d.year_number, COUNT(*) AS loans, ROUND(AVG(l.default_flag), 4) AS default_rate
    FROM fact_loan l
    JOIN dim_date d ON d.date_key = l.issue_date_key
    JOIN dim_loan_purpose p ON p.purpose_key = l.purpose_key
    GROUP BY p.purpose_name, d.year_number
)
ORDER BY purpose_name, year_number;

-- 4. Risk segment matrix: FICO band x DTI band (source for the BI heat map).
SELECT
    CASE WHEN fico_mid < 660 THEN '1: <660'
         WHEN fico_mid < 700 THEN '2: 660-699'
         WHEN fico_mid < 740 THEN '3: 700-739'
         ELSE '4: 740+' END AS fico_band,
    CASE WHEN debt_to_income IS NULL THEN '0: Unknown'
         WHEN debt_to_income < 10 THEN '1: <10'
         WHEN debt_to_income < 20 THEN '2: 10-19'
         WHEN debt_to_income < 30 THEN '3: 20-29'
         ELSE '4: 30+' END AS dti_band,
    COUNT(*) AS loans,
    ROUND(AVG(default_flag), 4) AS default_rate
FROM fact_loan
GROUP BY
    CASE WHEN fico_mid < 660 THEN '1: <660'
         WHEN fico_mid < 700 THEN '2: 660-699'
         WHEN fico_mid < 740 THEN '3: 700-739'
         ELSE '4: 740+' END,
    CASE WHEN debt_to_income IS NULL THEN '0: Unknown'
         WHEN debt_to_income < 10 THEN '1: <10'
         WHEN debt_to_income < 20 THEN '2: 10-19'
         WHEN debt_to_income < 30 THEN '3: 20-29'
         ELSE '4: 30+' END
ORDER BY fico_band, dti_band;

-- 5. Daily fraud rate with a 7-day rolling rate for monitoring.
--    The rolling rate is a ratio of rolling sums, not an average of daily rates,
--    so low-volume days do not get equal weight.
SELECT
    calendar_date,
    transactions,
    fraud_count,
    ROUND(fraud_count / transactions, 4) AS fraud_rate,
    ROUND(SUM(fraud_count) OVER w / SUM(transactions) OVER w, 4) AS fraud_rate_7d,
    ROUND(SUM(fraud_amount) OVER w, 2) AS fraud_amount_7d
FROM (
    SELECT d.calendar_date,
           COUNT(*) AS transactions,
           SUM(t.fraud_flag) AS fraud_count,
           SUM(CASE WHEN t.fraud_flag = 1 THEN t.transaction_amount ELSE 0 END) AS fraud_amount
    FROM fact_transaction t
    JOIN dim_date d ON d.date_key = t.transaction_date_key
    GROUP BY d.calendar_date
)
WINDOW w AS (ORDER BY calendar_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
ORDER BY calendar_date;

-- 6. Fraud rate by transaction amount decile (NTILE).
SELECT amount_decile,
       MIN(transaction_amount) AS min_amount,
       MAX(transaction_amount) AS max_amount,
       COUNT(*) AS transactions,
       ROUND(AVG(fraud_flag), 4) AS fraud_rate
FROM (
    SELECT transaction_amount, fraud_flag,
           NTILE(10) OVER (ORDER BY transaction_amount) AS amount_decile
    FROM fact_transaction
)
GROUP BY amount_decile
ORDER BY amount_decile;

-- 7. Does velocity separate fraud? Fraud rate by 24h transaction-count bucket per product.
SELECT
    p.product_code,
    CASE WHEN t.tx_count_24h = 1 THEN '1'
         WHEN t.tx_count_24h <= 3 THEN '2-3'
         WHEN t.tx_count_24h <= 10 THEN '4-10'
         ELSE '11+' END AS tx_count_24h_bucket,
    COUNT(*) AS transactions,
    ROUND(AVG(t.fraud_flag), 4) AS fraud_rate
FROM fact_transaction t
JOIN dim_product p ON p.product_key = t.product_key
GROUP BY
    p.product_code,
    CASE WHEN t.tx_count_24h = 1 THEN '1'
         WHEN t.tx_count_24h <= 3 THEN '2-3'
         WHEN t.tx_count_24h <= 10 THEN '4-10'
         ELSE '11+' END
ORDER BY p.product_code, MIN(t.tx_count_24h);

-- 8. Highest-risk cards: cards with at least 5 transactions, ranked by fraud count
--    within each card network, keeping the top 10 per network.
SELECT *
FROM (
    SELECT
        c.card_network,
        c.card_natural_key,
        COUNT(*) AS transactions,
        SUM(t.fraud_flag) AS fraud_count,
        MAX(t.tx_count_24h) AS peak_tx_count_24h,
        DENSE_RANK() OVER (PARTITION BY c.card_network ORDER BY SUM(t.fraud_flag) DESC) AS network_rank
    FROM fact_transaction t
    JOIN dim_card c ON c.card_key = t.card_key
    GROUP BY c.card_network, c.card_natural_key
    HAVING COUNT(*) >= 5
)
WHERE network_rank <= 10
ORDER BY card_network, network_rank;
