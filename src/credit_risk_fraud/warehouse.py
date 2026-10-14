import os
import re
from pathlib import Path

import pandas as pd

# Staging column -> processed Parquet column. Order defines the INSERT column order.
LOAN_STAGING_COLUMNS = {
    "loan_id": "id",
    "issue_date": "issue_date",
    "loan_status": "loan_status",
    "default_flag": "default_flag",
    "purpose": "purpose",
    "addr_state": "addr_state",
    "grade": "grade",
    "sub_grade": "sub_grade",
    "home_ownership": "home_ownership",
    "verification_status": "verification_status",
    "application_type": "application_type",
    "emp_length_years": "emp_length_years",
    "credit_history_bucket": "credit_history_bucket",
    "term_months": "term_months",
    "loan_amnt": "loan_amnt",
    "int_rate": "int_rate",
    "installment": "installment",
    "annual_inc": "annual_inc",
    "dti": "dti",
    "fico_mid": "fico_mid",
    "credit_history_months": "credit_history_months",
    "loan_to_income": "loan_to_income",
    "installment_to_income": "installment_to_income",
    "revol_util": "revol_util",
    "inq_last_6mths": "inq_last_6mths",
    "delinq_2yrs": "delinq_2yrs",
    "pub_rec_bankruptcies": "pub_rec_bankruptcies",
}

TRANSACTION_STAGING_COLUMNS = {
    "transaction_id": "TransactionID",
    "transaction_dt": "TransactionDT",
    "fraud_flag": "fraud_flag",
    "transaction_amount": "transaction_amount",
    "product_code": "ProductCD",
    "card_natural_key": "card_key",
    "card1": "card1",
    "card_network": "card4",
    "card_type": "card6",
    "billing_region": "addr1",
    "email_domain": "P_emaildomain",
    "device_type": "DeviceType",
    "tx_count_1h": "tx_count_1h",
    "tx_amount_1h": "tx_amount_1h",
    "tx_count_24h": "tx_count_24h",
    "tx_amount_24h": "tx_amount_24h",
    "tx_count_7d": "tx_count_7d",
    "tx_amount_7d": "tx_amount_7d",
    "amount_to_7d_mean": "amount_to_7d_mean",
    "seconds_since_prev_tx": "seconds_since_prev_tx",
}


def connect():
    """Connect with python-oracledb thin mode (no Oracle client install needed)."""
    import oracledb

    return oracledb.connect(
        user=os.environ.get("ORACLE_USER", "CREDIT_DW"),
        password=os.environ.get("APP_USER_PASSWORD", "CreditDw123"),
        dsn=os.environ.get("ORACLE_DSN", "localhost:1521/FREEPDB1"),
    )


def split_sql_statements(script: str) -> list[str]:
    """Split a plain-SQL script (no PL/SQL blocks) into statements, dropping comments."""
    without_comments = re.sub(r"--[^\n]*", "", script)
    return [statement.strip() for statement in without_comments.split(";") if statement.strip()]


def run_sql_script(connection, path: str | Path) -> None:
    with connection.cursor() as cursor:
        for statement in split_sql_statements(Path(path).read_text(encoding="utf-8")):
            cursor.execute(statement)
    connection.commit()


def staging_frame(data: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
    """Select and rename Parquet columns for a staging table; absent optional columns become null."""
    frame = pd.DataFrame(
        {target: data[source] if source in data.columns else None for target, source in column_map.items()},
        index=data.index,
    )
    # Oracle binds need plain Python values with None for missing, not NaN/NaT/pd.NA.
    return frame.astype(object).where(frame.notna(), None)


def load_staging(connection, table: str, frame: pd.DataFrame, batch_size: int = 50_000) -> int:
    import oracledb

    columns = list(frame.columns)
    placeholders = ", ".join(f":{position}" for position in range(1, len(columns) + 1))
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    input_sizes = [_bind_type(frame[name], oracledb) for name in columns]
    with connection.cursor() as cursor:
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start:start + batch_size]
            # Explicit bind types stop a leading None from fixing a column's type as text.
            cursor.setinputsizes(*input_sizes)
            cursor.executemany(sql, list(batch.itertuples(index=False, name=None)))
    connection.commit()
    return len(frame)


def _bind_type(column: pd.Series, oracledb):
    values = column.dropna()
    if values.empty:
        return oracledb.DB_TYPE_VARCHAR
    sample = values.iloc[0]
    if isinstance(sample, str):
        return int(values.map(len).max())
    if hasattr(sample, "year"):
        return oracledb.DB_TYPE_DATE
    return oracledb.DB_TYPE_NUMBER


def run_checks(connection, path: str | Path) -> list[tuple[str, float, str]]:
    """Run data quality queries; each returns (check_name, observed_value, status)."""
    results = []
    with connection.cursor() as cursor:
        for statement in split_sql_statements(Path(path).read_text(encoding="utf-8")):
            cursor.execute(statement)
            results.extend(cursor.fetchall())
    return results
