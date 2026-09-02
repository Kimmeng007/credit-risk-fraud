from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from credit_risk_fraud.spark_pipeline import process_credit_data, process_fraud_data

CREDIT_CSV = (
    b"id,desc,annual_inc,dti,fico_range_low,fico_range_high,loan_status\n"
    b'1,"single line desc",50000,15.5,700,704,Fully Paid\n'
    b'2,"multi\nline desc",60000,20.0,680,684,Charged Off\n'
    b'3,"another",,25.0,720,724,Current\n'
)

FRAUD_TRANSACTION_CSV = (
    b"TransactionID,TransactionDT,isFraud,TransactionAmt,card1\n"
    b"1,1000,0,50.00,100\n"
    b"2,2000,1,75.00,100\n"
    b"3,3000,0,20.00,200\n"
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

    assert set(rows) == {1, 2, 3}
    assert rows[2]["default_flag"] == 1
    assert rows[1]["default_flag"] == 0
    assert rows[3]["fico_mid"] == pytest.approx(722.0)
    assert rows[3]["income_missing"] == 1
    assert rows[1]["income_missing"] == 0


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
    # rangeBetween(-24h, 0) is backward-looking: row 1 (DT=1000) only sees itself,
    # row 2 (DT=2000) sees both prior card1=100 transactions.
    assert rows[1]["tx_count_24h"] == 1
    assert rows[2]["tx_count_24h"] == 2
    assert rows[3]["tx_count_24h"] == 1
    assert rows[1]["tx_amount_24h"] == pytest.approx(50.0)
    assert rows[2]["tx_amount_24h"] == pytest.approx(125.0)
