CREATE TABLE dim_time (
    time_key NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    calendar_date DATE NOT NULL,
    year_number NUMBER(4) NOT NULL,
    month_number NUMBER(2) NOT NULL,
    quarter_number NUMBER(1) NOT NULL
);

CREATE TABLE dim_loan_purpose (
    purpose_key NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    purpose_name VARCHAR2(100) NOT NULL UNIQUE
);

CREATE TABLE fact_loan (
    loan_key NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    time_key NUMBER NOT NULL REFERENCES dim_time(time_key),
    purpose_key NUMBER REFERENCES dim_loan_purpose(purpose_key),
    source_loan_id VARCHAR2(40) NOT NULL UNIQUE,
    annual_income NUMBER(18,2),
    debt_to_income NUMBER(8,3),
    fico_mid NUMBER(6,2),
    default_flag NUMBER(1) NOT NULL CHECK (default_flag IN (0, 1))
);

CREATE TABLE fact_transaction (
    transaction_key NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    time_key NUMBER REFERENCES dim_time(time_key),
    source_transaction_id NUMBER NOT NULL UNIQUE,
    card_key VARCHAR2(100),
    transaction_amount NUMBER(18,2),
    tx_count_24h NUMBER,
    tx_amount_24h NUMBER(18,2),
    fraud_flag NUMBER(1) NOT NULL CHECK (fraud_flag IN (0, 1))
);

CREATE INDEX ix_fact_loan_time ON fact_loan(time_key);
CREATE INDEX ix_fact_transaction_time ON fact_transaction(time_key);