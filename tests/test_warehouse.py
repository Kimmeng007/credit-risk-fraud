import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_fraud.warehouse import (
    LOAN_STAGING_COLUMNS,
    split_sql_statements,
    staging_frame,
)

SQL_DIR = Path(__file__).resolve().parents[1] / "sql"


def test_split_sql_statements_drops_comments_and_blank_statements():
    script = """
    -- header comment; with a semicolon
    SELECT 1 FROM dual;  -- trailing comment

    SELECT 2
    FROM dual;
    ;
    """

    assert split_sql_statements(script) == ["SELECT 1 FROM dual", "SELECT 2\n    FROM dual"]


def test_schema_creates_both_star_schemas():
    statements = split_sql_statements((SQL_DIR / "01_schema.sql").read_text(encoding="utf-8"))
    created = {s.split()[2].lower() for s in statements if s.upper().startswith("CREATE TABLE")}

    assert {"fact_loan", "fact_transaction", "dim_date", "stg_loan", "stg_transaction"} <= created
    # Every created table is dropped first so the script can rebuild the warehouse.
    dropped = {s.split()[4].lower() for s in statements if s.upper().startswith("DROP TABLE")}
    assert created == dropped


def test_load_and_check_scripts_split_cleanly():
    for name in ("02_analytical_queries.sql", "03_load_warehouse.sql", "04_data_quality_checks.sql"):
        statements = split_sql_statements((SQL_DIR / name).read_text(encoding="utf-8"))
        assert statements, name
        assert all(s.split()[0].upper() in {"SELECT", "INSERT"} for s in statements), name


def test_bi_views_match_export_list():
    from importlib import util

    statements = split_sql_statements((SQL_DIR / "05_bi_views.sql").read_text(encoding="utf-8"))
    created = {s.split()[4].lower() for s in statements}
    assert all(s.upper().startswith("CREATE OR REPLACE VIEW") for s in statements)

    spec = util.spec_from_file_location("export_bi", SQL_DIR.parent / "scripts" / "export_bi_extracts.py")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert created == set(module.BI_VIEWS)


def test_staging_frame_renames_and_converts_missing_values_to_none():
    data = pd.DataFrame(
        {
            "id": ["1", "2"],
            "issue_date": [datetime.date(2015, 12, 1), None],
            "annual_inc": [50000.0, np.nan],
            "default_flag": np.array([0, 1], dtype="int32"),
        }
    )

    frame = staging_frame(data, LOAN_STAGING_COLUMNS)

    assert list(frame.columns) == list(LOAN_STAGING_COLUMNS)
    assert frame.loc[0, "loan_id"] == "1"
    assert frame.loc[1, "annual_inc"] is None
    assert frame.loc[1, "issue_date"] is None
    assert frame.loc[1, "default_flag"] == 1
    # Columns absent from the source become all-null instead of raising.
    assert frame["purpose"].isna().all()
