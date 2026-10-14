from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from credit_risk_fraud.spark_pipeline import process_credit_data, process_fraud_data

CREDIT_CSV = (
    b"id,desc,loan_amnt,term,installment,emp_length,annual_inc,dti,issue_d,earliest_cr_line,"
    b"fico_range_low,fico_range_high,loan_status,total_pymnt\n"
    b'1,"single line desc",12000,36 months,400.0,10+ years,48000,15.5,Dec-2015,Dec-2005,'
    b"700,704,Fully Paid,13000\n"
    b'2,"multi\nline desc",6000,60 months,150.0,< 1 year,60000,-1,Jan-2014,Jul-2012,'
    b"680,684,Charged Off,2000\n"
    b'3,"another",5000,36 months,170.0,,,25.0,Mar-2016,Jan-1990,'
    b"720,724,Does not meet the credit policy. Status:Fully Paid,5200\n"
    b'4,"still open",8000,36 months,260.0,3 years,70000,10.0,Jun-2018,Jan-2000,'
    b"710,714,Current,1000\n"
    b'5,"late",8000,36 months,260.0,3 years,70000,10.0,Jun-2018,Jan-2000,'
    b"710,714,Late (31-120 days),1000\n"
)

# Card 100 at DT 1000, 2000 and 1000 + 2 days: the third transaction falls outside the
# 1h and 24h windows of the first two but inside the 7d window.
FRAUD_TRANSACTION_CSV = (
    b"TransactionID,TransactionDT,isFraud,TransactionAmt,card1,addr1\n"
    b"1,1000,0,50.00,100,10\n"
    b"2,2000,1,75.00,100,10\n"
    b"3,3000,0,20.00,200,10\n"
    b"4,173800,0,25.00,100,10\n"
    b"5,2500,0,99.00,100,99\n"
)


@pytest.fixture
def spark_paths(tmp_path):
    credit_dir = tmp_path / "raw" / "lending-club"
    credit_dir.mkdir(parents=True)
    (credit_dir / "accepted.csv").write_bytes(CREDIT_CSV)

    fraud_dir = tmp_path / "raw" / "ieee-cis"
    fraud_dir.mkdir(parents=True)
    (fraud_dir / "train_transaction.csv").write_bytes(FRAUD_TRANSACTION_CSV)

    output_dir = tmp_path / "processed"
    return credit_dir, fraud_dir, output_dir


def _read_parquet(spark, path: Path):
    return spark.read.parquet(str(path))


def test_process_credit_data_handles_embedded_newlines_and_defaults(spark_paths):
    from pyspark.sql import SparkSession

    credit_dir, _, output_dir = spark_paths
    process_credit_data(credit_dir, output_dir / "credit")

    spark = SparkSession.builder.appName("test-read").getOrCreate()
    try:
        rows = {row["id"]: row for row in _read_parquet(spark, output_dir / "credit").collect()}
    finally:
        spark.stop()

    # Open loans (Current, Late) have no final outcome and must be excluded, not labeled 0.
    assert set(rows) == {1, 2, 3}
    assert rows[2]["default_flag"] == 1
    assert rows[1]["default_flag"] == 0
    assert rows[3]["default_flag"] == 0
    assert rows[3]["fico_mid"] == pytest.approx(722.0)
    assert rows[3]["income_missing"] == 1
    assert rows[1]["income_missing"] == 0

    assert rows[1]["credit_history_months"] == 120
    assert rows[1]["credit_history_bucket"] == "10-20y"
    assert rows[2]["credit_history_bucket"] == "<3y"
    assert rows[3]["credit_history_bucket"] == "20y+"
    assert rows[1]["term_months"] == 36
    assert rows[2]["term_months"] == 60
    assert rows[1]["emp_length_years"] == 10
    assert rows[2]["emp_length_years"] == 0
    assert rows[3]["emp_length_years"] is None
    assert rows[1]["loan_to_income"] == pytest.approx(0.25)
    assert rows[1]["installment_to_income"] == pytest.approx(0.1)
    assert rows[3]["loan_to_income"] is None
    assert rows[2]["dti"] is None  # negative DTI is Lending Club's missing-value code

    # Post-origination fields would leak the outcome and must not be carried forward.
    assert "total_pymnt" not in rows[1].asDict()
    assert "desc" not in rows[1].asDict()


def test_process_fraud_data_computes_velocity_features(spark_paths):
    from pyspark.sql import SparkSession

    _, fraud_dir, output_dir = spark_paths
    process_fraud_data(fraud_dir, output_dir / "fraud")

    spark = SparkSession.builder.appName("test-read").getOrCreate()
    try:
        rows = {
            row["TransactionID"]: row for row in _read_parquet(spark, output_dir / "fraud").collect()
        }
    finally:
        spark.stop()

    assert rows[2]["fraud_flag"] == 1
    # Windows are backward-looking: row 1 (DT=1000) only sees itself,
    # row 2 (DT=2000) sees both card 100 / addr 10 transactions so far.
    assert rows[1]["tx_count_24h"] == 1
    assert rows[2]["tx_count_24h"] == 2
    assert rows[3]["tx_count_24h"] == 1
    assert rows[1]["tx_amount_24h"] == pytest.approx(50.0)
    assert rows[2]["tx_amount_24h"] == pytest.approx(125.0)
    assert rows[2]["tx_count_1h"] == 2

    # Row 5 shares card1 but has a different addr1, so it is a different card_key.
    assert rows[5]["card_key"] != rows[2]["card_key"]
    assert rows[5]["tx_count_7d"] == 1

    # Row 4 is two days later: outside 24h, inside 7d.
    assert rows[4]["tx_count_24h"] == 1
    assert rows[4]["tx_count_7d"] == 3
    assert rows[4]["tx_amount_7d"] == pytest.approx(150.0)
    assert rows[4]["amount_to_7d_mean"] == pytest.approx(0.5)
    assert rows[4]["seconds_since_prev_tx"] == 171800
    assert rows[1]["seconds_since_prev_tx"] is None
