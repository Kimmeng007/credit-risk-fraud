-- Credit risk & fraud warehouse: two star schemas sharing a conformed date dimension.
-- Rebuilds from scratch; staging tables receive Spark Parquet output, then
-- 03_load_warehouse.sql populates dimensions and facts with set-based SQL.

DROP TABLE IF EXISTS fact_loan PURGE;
DROP TABLE IF EXISTS fact_transaction PURGE;
DROP TABLE IF EXISTS dim_date PURGE;
DROP TABLE IF EXISTS dim_loan_purpose PURGE;
DROP TABLE IF EXISTS dim_region PURGE;
DROP TABLE IF EXISTS dim_credit_grade PURGE;
DROP TABLE IF EXISTS dim_borrower_profile PURGE;
DROP TABLE IF EXISTS dim_card PURGE;
DROP TABLE IF EXISTS dim_product PURGE;
DROP TABLE IF EXISTS stg_loan PURGE;
DROP TABLE IF EXISTS stg_transaction PURGE;

-- ---------------------------------------------------------------------------
-- Staging (one row per processed Parquet record, no constraints beyond types)
-- ---------------------------------------------------------------------------

CREATE TABLE stg_loan (
    loan_id                VARCHAR2(20),
    issue_date             DATE,
    loan_status            VARCHAR2(60),
    default_flag           NUMBER(1),
    purpose                VARCHAR2(40),
    addr_state             VARCHAR2(2),
    grade                  VARCHAR2(1),
    sub_grade              VARCHAR2(2),
    home_ownership         VARCHAR2(20),
    verification_status    VARCHAR2(20),
    application_type       VARCHAR2(20),
    emp_length_years       NUMBER(2),
    credit_history_bucket  VARCHAR2(10),
    term_months            NUMBER(3),
    loan_amnt              NUMBER(12,2),
    int_rate               NUMBER(6,3),
    installment            NUMBER(12,2),
    annual_inc             NUMBER(14,2),
    dti                    NUMBER(10,3),
    fico_mid               NUMBER(6,1),
    credit_history_months  NUMBER(5),
    loan_to_income         NUMBER,
    installment_to_income  NUMBER,
    revol_util             NUMBER(8,2),
    inq_last_6mths         NUMBER(4),
    delinq_2yrs            NUMBER(4),
    pub_rec_bankruptcies   NUMBER(4),
    -- Banded in the database so the dimension and fact loads share one definition.
    emp_length_band        VARCHAR2(10) GENERATED ALWAYS AS (
        CASE
            WHEN emp_length_years IS NULL THEN 'Unknown'
            WHEN emp_length_years < 1 THEN '<1y'
            WHEN emp_length_years <= 3 THEN '1-3y'
            WHEN emp_length_years <= 6 THEN '4-6y'
            WHEN emp_length_years <= 9 THEN '7-9y'
            ELSE '10y+'
        END
    ) VIRTUAL
);

CREATE TABLE stg_transaction (
    transaction_id         NUMBER(12),
    transaction_dt         NUMBER(12),
    fraud_flag             NUMBER(1),
    transaction_amount     NUMBER(14,3),
    product_code           VARCHAR2(5),
    card_natural_key       VARCHAR2(100),
    card1                  NUMBER(8),
    card_network           VARCHAR2(30),
    card_type              VARCHAR2(30),
    billing_region         NUMBER(6),
    email_domain           VARCHAR2(60),
    device_type            VARCHAR2(20),
    tx_count_1h            NUMBER(8),
    tx_amount_1h           NUMBER(16,3),
    tx_count_24h           NUMBER(8),
    tx_amount_24h          NUMBER(16,3),
    tx_count_7d            NUMBER(8),
    tx_amount_7d           NUMBER(16,3),
    amount_to_7d_mean      NUMBER,
    seconds_since_prev_tx  NUMBER(12)
);

-- ---------------------------------------------------------------------------
-- Conformed dimension
-- ---------------------------------------------------------------------------

CREATE TABLE dim_date (
    date_key          NUMBER(8) PRIMARY KEY,          -- YYYYMMDD smart key
    calendar_date     DATE NOT NULL UNIQUE,
    year_number       NUMBER(4) NOT NULL,
    quarter_number    NUMBER(1) NOT NULL,
    month_number      NUMBER(2) NOT NULL,
    month_start_date  DATE NOT NULL,
    month_name        VARCHAR2(9) NOT NULL,
    day_of_week       VARCHAR2(9) NOT NULL,
    is_weekend        NUMBER(1) NOT NULL CHECK (is_weekend IN (0, 1))
);

-- ---------------------------------------------------------------------------
-- Credit star
-- ---------------------------------------------------------------------------

CREATE TABLE dim_loan_purpose (
    purpose_key   NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    purpose_name  VARCHAR2(40) NOT NULL UNIQUE
);

CREATE TABLE dim_region (
    region_key  NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    state_code  VARCHAR2(7) NOT NULL UNIQUE
);

-- Lending Club's own grade is kept for BI segmentation, not used as a model feature.
CREATE TABLE dim_credit_grade (
    grade_key  NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    grade      VARCHAR2(7) NOT NULL,
    sub_grade  VARCHAR2(7) NOT NULL UNIQUE
);

-- Junk dimension: the public data has no persistent customer id, so the borrower
-- is described by the combination of low-cardinality applicant attributes.
CREATE TABLE dim_borrower_profile (
    profile_key            NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    home_ownership         VARCHAR2(20) NOT NULL,
    verification_status    VARCHAR2(20) NOT NULL,
    application_type       VARCHAR2(20) NOT NULL,
    emp_length_band        VARCHAR2(10) NOT NULL,
    credit_history_bucket  VARCHAR2(10) NOT NULL,
    CONSTRAINT uq_borrower_profile UNIQUE (
        home_ownership, verification_status, application_type,
        emp_length_band, credit_history_bucket
    )
);

CREATE TABLE fact_loan (
    loan_id                VARCHAR2(20) PRIMARY KEY,
    issue_date_key         NUMBER(8) NOT NULL REFERENCES dim_date(date_key),
    purpose_key            NUMBER NOT NULL REFERENCES dim_loan_purpose(purpose_key),
    region_key             NUMBER NOT NULL REFERENCES dim_region(region_key),
    grade_key              NUMBER NOT NULL REFERENCES dim_credit_grade(grade_key),
    profile_key            NUMBER NOT NULL REFERENCES dim_borrower_profile(profile_key),
    term_months            NUMBER(3) NOT NULL,
    loan_amount            NUMBER(12,2),
    interest_rate          NUMBER(6,3),
    installment            NUMBER(12,2),
    annual_income          NUMBER(14,2),
    debt_to_income         NUMBER(10,3),
    fico_mid               NUMBER(6,1),
    credit_history_months  NUMBER(5),
    loan_to_income         NUMBER,
    installment_to_income  NUMBER,
    revol_util             NUMBER(8,2),
    inq_last_6mths         NUMBER(4),
    delinq_2yrs            NUMBER(4),
    pub_rec_bankruptcies   NUMBER(4),
    loan_status            VARCHAR2(60) NOT NULL,
    default_flag           NUMBER(1) NOT NULL CHECK (default_flag IN (0, 1))
);

CREATE INDEX ix_fact_loan_date ON fact_loan(issue_date_key);
CREATE INDEX ix_fact_loan_purpose ON fact_loan(purpose_key);
CREATE INDEX ix_fact_loan_region ON fact_loan(region_key);
CREATE INDEX ix_fact_loan_grade ON fact_loan(grade_key);
CREATE INDEX ix_fact_loan_profile ON fact_loan(profile_key);

-- ---------------------------------------------------------------------------
-- Fraud star
-- ---------------------------------------------------------------------------

CREATE TABLE dim_product (
    product_key   NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    product_code  VARCHAR2(7) NOT NULL UNIQUE
);

-- card_natural_key = card1|addr1|P_emaildomain, the same key used for Spark velocity features.
CREATE TABLE dim_card (
    card_key          NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    card_natural_key  VARCHAR2(100) NOT NULL UNIQUE,
    card1             NUMBER(8),
    card_network      VARCHAR2(30) NOT NULL,
    card_type         VARCHAR2(30) NOT NULL,
    billing_region    NUMBER(6),
    email_domain      VARCHAR2(60) NOT NULL
);

CREATE TABLE fact_transaction (
    transaction_id         NUMBER(12) PRIMARY KEY,
    transaction_date_key   NUMBER(8) NOT NULL REFERENCES dim_date(date_key),
    card_key               NUMBER NOT NULL REFERENCES dim_card(card_key),
    product_key            NUMBER NOT NULL REFERENCES dim_product(product_key),
    transaction_ts         DATE NOT NULL,
    transaction_dt         NUMBER(12) NOT NULL,
    transaction_amount     NUMBER(14,3),
    device_type            VARCHAR2(20) NOT NULL,
    tx_count_1h            NUMBER(8),
    tx_amount_1h           NUMBER(16,3),
    tx_count_24h           NUMBER(8),
    tx_amount_24h          NUMBER(16,3),
    tx_count_7d            NUMBER(8),
    tx_amount_7d           NUMBER(16,3),
    amount_to_7d_mean      NUMBER,
    seconds_since_prev_tx  NUMBER(12),
    fraud_flag             NUMBER(1) NOT NULL CHECK (fraud_flag IN (0, 1))
);

CREATE INDEX ix_fact_tx_date ON fact_transaction(transaction_date_key);
CREATE INDEX ix_fact_tx_card ON fact_transaction(card_key);
CREATE INDEX ix_fact_tx_product ON fact_transaction(product_key);
