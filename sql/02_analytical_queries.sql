SELECT
    t.year_number,
    t.month_number,
    p.purpose_name,
    COUNT(*) AS loan_count,
    AVG(l.default_flag) AS default_rate,
    AVG(l.fico_mid) AS average_fico
FROM fact_loan l
JOIN dim_time t ON t.time_key = l.time_key
LEFT JOIN dim_loan_purpose p ON p.purpose_key = l.purpose_key
GROUP BY t.year_number, t.month_number, p.purpose_name
ORDER BY t.year_number, t.month_number, p.purpose_name;

SELECT
    t.calendar_date,
    COUNT(*) AS transaction_count,
    SUM(tx.fraud_flag) AS fraud_count,
    AVG(tx.fraud_flag) AS fraud_rate,
    SUM(tx.transaction_amount) AS transaction_amount
FROM fact_transaction tx
JOIN dim_time t ON t.time_key = tx.time_key
GROUP BY t.calendar_date
ORDER BY t.calendar_date;

SELECT *
FROM (
    SELECT
        p.purpose_name,
        t.year_number,
        AVG(l.default_flag) AS default_rate,
        RANK() OVER (PARTITION BY t.year_number ORDER BY AVG(l.default_flag) DESC) AS purpose_rank
    FROM fact_loan l
    JOIN dim_time t ON t.time_key = l.time_key
    JOIN dim_loan_purpose p ON p.purpose_key = l.purpose_key
    GROUP BY p.purpose_name, t.year_number
)
WHERE purpose_rank <= 5;