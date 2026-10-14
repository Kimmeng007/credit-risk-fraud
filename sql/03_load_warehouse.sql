-- Populate dimensions and facts from the staging tables (set-based ELT).
-- Missing categorical values map to an explicit 'Unknown' member so every
-- fact row joins to every dimension.

-- IEEE-CIS TransactionDT is a seconds offset from an undisclosed reference time.
-- 2017-12-01 is the commonly used assumption; calendar attributes for fraud are
-- therefore approximate, while ordering and window features are exact.

-- Conformed date dimension: one row per day covering both datasets.
INSERT INTO dim_date (
    date_key, calendar_date, year_number, quarter_number, month_number,
    month_start_date, month_name, day_of_week, is_weekend
)
SELECT
    TO_NUMBER(TO_CHAR(d, 'YYYYMMDD')),
    d,
    EXTRACT(YEAR FROM d),
    TO_NUMBER(TO_CHAR(d, 'Q')),
    EXTRACT(MONTH FROM d),
    TRUNC(d, 'MM'),
    TRIM(TO_CHAR(d, 'Month', 'NLS_DATE_LANGUAGE=ENGLISH')),
    TRIM(TO_CHAR(d, 'Day', 'NLS_DATE_LANGUAGE=ENGLISH')),
    CASE WHEN TO_CHAR(d, 'DY', 'NLS_DATE_LANGUAGE=ENGLISH') IN ('SAT', 'SUN') THEN 1 ELSE 0 END
FROM (
    SELECT DATE '2007-01-01' + LEVEL - 1 AS d
    FROM dual
    CONNECT BY LEVEL <= DATE '2019-12-31' - DATE '2007-01-01' + 1
);

-- Credit dimensions
INSERT INTO dim_loan_purpose (purpose_name)
SELECT DISTINCT NVL(purpose, 'Unknown') FROM stg_loan;

INSERT INTO dim_region (state_code)
SELECT DISTINCT NVL(addr_state, 'Unknown') FROM stg_loan;

INSERT INTO dim_credit_grade (grade, sub_grade)
SELECT DISTINCT NVL(grade, 'Unknown'), NVL(sub_grade, 'Unknown') FROM stg_loan;

INSERT INTO dim_borrower_profile (
    home_ownership, verification_status, application_type,
    emp_length_band, credit_history_bucket
)
SELECT DISTINCT
    NVL(home_ownership, 'Unknown'),
    NVL(verification_status, 'Unknown'),
    NVL(application_type, 'Unknown'),
    emp_length_band,
    NVL(credit_history_bucket, 'Unknown')
FROM stg_loan;

INSERT INTO fact_loan (
    loan_id, issue_date_key, purpose_key, region_key, grade_key, profile_key,
    term_months, loan_amount, interest_rate, installment, annual_income, debt_to_income,
    fico_mid, credit_history_months, loan_to_income, installment_to_income, revol_util,
    inq_last_6mths, delinq_2yrs, pub_rec_bankruptcies, loan_status, default_flag
)
SELECT
    s.loan_id,
    TO_NUMBER(TO_CHAR(s.issue_date, 'YYYYMMDD')),
    p.purpose_key,
    r.region_key,
    g.grade_key,
    b.profile_key,
    s.term_months,
    s.loan_amnt,
    s.int_rate,
    s.installment,
    s.annual_inc,
    s.dti,
    s.fico_mid,
    s.credit_history_months,
    s.loan_to_income,
    s.installment_to_income,
    s.revol_util,
    s.inq_last_6mths,
    s.delinq_2yrs,
    s.pub_rec_bankruptcies,
    s.loan_status,
    s.default_flag
FROM stg_loan s
JOIN dim_loan_purpose p ON p.purpose_name = NVL(s.purpose, 'Unknown')
JOIN dim_region r ON r.state_code = NVL(s.addr_state, 'Unknown')
JOIN dim_credit_grade g ON g.sub_grade = NVL(s.sub_grade, 'Unknown')
JOIN dim_borrower_profile b
    ON b.home_ownership = NVL(s.home_ownership, 'Unknown')
    AND b.verification_status = NVL(s.verification_status, 'Unknown')
    AND b.application_type = NVL(s.application_type, 'Unknown')
    AND b.emp_length_band = s.emp_length_band
    AND b.credit_history_bucket = NVL(s.credit_history_bucket, 'Unknown');

-- Fraud dimensions
INSERT INTO dim_product (product_code)
SELECT DISTINCT NVL(product_code, 'Unknown') FROM stg_transaction;

-- A card key can see more than one network/type label in the raw data; keep the
-- most frequent one so the dimension stays one row per card.
INSERT INTO dim_card (card_natural_key, card1, card_network, card_type, billing_region, email_domain)
SELECT card_natural_key, card1, card_network, card_type, billing_region, email_domain
FROM (
    SELECT
        card_natural_key,
        card1,
        NVL(card_network, 'Unknown') AS card_network,
        NVL(card_type, 'Unknown') AS card_type,
        billing_region,
        NVL(email_domain, 'Unknown') AS email_domain,
        ROW_NUMBER() OVER (
            PARTITION BY card_natural_key
            ORDER BY COUNT(*) DESC, NVL(card_network, 'Unknown'), NVL(card_type, 'Unknown')
        ) AS label_rank
    FROM stg_transaction
    GROUP BY card_natural_key, card1, card_network, card_type, billing_region, email_domain
)
WHERE label_rank = 1;

INSERT INTO fact_transaction (
    transaction_id, transaction_date_key, card_key, product_key, transaction_ts,
    transaction_dt, transaction_amount, device_type, tx_count_1h, tx_amount_1h,
    tx_count_24h, tx_amount_24h, tx_count_7d, tx_amount_7d, amount_to_7d_mean,
    seconds_since_prev_tx, fraud_flag
)
SELECT
    s.transaction_id,
    TO_NUMBER(TO_CHAR(DATE '2017-12-01' + s.transaction_dt / 86400, 'YYYYMMDD')),
    c.card_key,
    p.product_key,
    DATE '2017-12-01' + s.transaction_dt / 86400,
    s.transaction_dt,
    s.transaction_amount,
    NVL(s.device_type, 'Unknown'),
    s.tx_count_1h,
    s.tx_amount_1h,
    s.tx_count_24h,
    s.tx_amount_24h,
    s.tx_count_7d,
    s.tx_amount_7d,
    s.amount_to_7d_mean,
    s.seconds_since_prev_tx,
    s.fraud_flag
FROM stg_transaction s
JOIN dim_card c ON c.card_natural_key = s.card_natural_key
JOIN dim_product p ON p.product_code = NVL(s.product_code, 'Unknown');
