from pathlib import Path

from .validation import require_real_dataset


def _spark_session(app_name: str):
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError("Install the Spark extra with `pip install -e .[spark]`.") from error
    return SparkSession.builder.appName(app_name).getOrCreate()


def process_credit_data(input_path: str | Path, output_path: str | Path) -> None:
    source = require_real_dataset(input_path, "Lending Club")
    spark = _spark_session("credit-risk-processing")
    try:
        from pyspark.sql import functions as F

        loans = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("escape", '"')
            .csv(str(source))
        )
        required = {"loan_status", "annual_inc", "dti", "fico_range_low", "fico_range_high"}
        missing = required.difference(loans.columns)
        if missing:
            raise ValueError(f"Lending Club input is missing columns: {sorted(missing)}")
        processed = (
            loans.withColumn("default_flag", F.when(F.col("loan_status").isin("Charged Off", "Default"), 1).otherwise(0))
            .withColumn("fico_range_low", F.col("fico_range_low").cast("double"))
            .withColumn("fico_range_high", F.col("fico_range_high").cast("double"))
            .withColumn("fico_mid", (F.col("fico_range_low") + F.col("fico_range_high")) / 2)
            .withColumn("annual_inc", F.col("annual_inc").cast("double"))
            .withColumn("dti", F.col("dti").cast("double"))
            .withColumn("income_missing", F.col("annual_inc").isNull().cast("int"))
            .dropna(subset=["fico_mid", "default_flag"])
        )
        processed.write.mode("overwrite").parquet(str(output_path))
    finally:
        spark.stop()


def process_fraud_data(input_path: str | Path, output_path: str | Path) -> None:
    source = require_real_dataset(input_path, "IEEE-CIS fraud")
    spark = _spark_session("fraud-processing")
    try:
        from pyspark.sql import Window
        from pyspark.sql import functions as F

        transaction_source = source / "train_transaction.csv" if source.is_dir() else source
        transaction_source = require_real_dataset(transaction_source, "IEEE-CIS transactions")
        transactions = spark.read.option("header", True).option("inferSchema", True).csv(str(transaction_source))
        required = {"TransactionID", "TransactionDT", "isFraud", "TransactionAmt"}
        missing = required.difference(transactions.columns)
        if missing:
            raise ValueError(f"IEEE-CIS input is missing columns: {sorted(missing)}")
        identity_source = source / "train_identity.csv" if source.is_dir() else None
        if identity_source and identity_source.exists():
            identity = spark.read.option("header", True).option("inferSchema", True).csv(str(identity_source))
            transactions = transactions.join(identity, on="TransactionID", how="left")
        partition_column = "card1" if "card1" in transactions.columns else "TransactionID"
        window = Window.partitionBy(partition_column).orderBy("TransactionDT").rangeBetween(-24 * 60 * 60, 0)
        processed = (
            transactions.withColumn("transaction_amount", F.col("TransactionAmt").cast("double"))
            .withColumn("fraud_flag", F.col("isFraud").cast("int"))
            .withColumn("tx_count_24h", F.count("TransactionID").over(window))
            .withColumn("tx_amount_24h", F.sum("transaction_amount").over(window))
        )
        processed.write.mode("overwrite").parquet(str(output_path))
    finally:
        spark.stop()