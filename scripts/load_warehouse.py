import argparse
import time

import pandas as pd

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.warehouse import (
    LOAN_STAGING_COLUMNS,
    TRANSACTION_STAGING_COLUMNS,
    connect,
    load_staging,
    run_checks,
    run_sql_script,
    staging_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the Oracle warehouse from processed Parquet and run data quality checks."
    )
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    args = parser.parse_args()
    paths = project_paths(args.root)
    sql_dir = paths.root / "sql"

    started = time.perf_counter()
    with connect() as connection:
        print("Rebuilding schema...")
        run_sql_script(connection, sql_dir / "01_schema.sql")

        loans = pd.read_parquet(paths.processed / "credit", columns=list(set(LOAN_STAGING_COLUMNS.values())))
        print(f"Staging {len(loans):,} loans...")
        load_staging(connection, "stg_loan", staging_frame(loans, LOAN_STAGING_COLUMNS))
        del loans

        transactions = pd.read_parquet(
            paths.processed / "fraud", columns=list(set(TRANSACTION_STAGING_COLUMNS.values()))
        )
        print(f"Staging {len(transactions):,} transactions...")
        load_staging(
            connection, "stg_transaction", staging_frame(transactions, TRANSACTION_STAGING_COLUMNS)
        )
        del transactions

        print("Loading dimensions and facts...")
        run_sql_script(connection, sql_dir / "03_load_warehouse.sql")

        print("Data quality checks:")
        results = run_checks(connection, sql_dir / "04_data_quality_checks.sql")
    for name, value, status in results:
        print(f"  [{status}] {name}: {value}")
    print(f"Done in {time.perf_counter() - started:.0f}s")
    if any(status == "FAIL" for _, _, status in results):
        raise SystemExit("Data quality checks failed; do not use this load downstream.")


if __name__ == "__main__":
    main()
