import os
from pathlib import Path

from .validation import require_real_dataset

# Only loans with a final outcome carry a usable label. "Current", "Late" and
# "In Grace Period" loans are still open, so treating them as non-defaults would
# bias the default rate downward.
DEFAULT_STATUSES = (
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
)
PAID_STATUSES = (
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
)

# Fields known at origination. Payment, recovery, hardship and settlement fields
# are recorded after the loan is issued and would leak the outcome into features.
CREDIT_ORIGINATION_COLUMNS = (
    "id",
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "application_type",
)

FRAUD_CARD_KEY_COLUMNS = ("card1", "addr1", "P_emaildomain")
FRAUD_VELOCITY_WINDOWS = {"1h": 60 * 60, "24h": 24 * 60 * 60, "7d": 7 * 24 * 60 * 60}


def _spark_session(app_name: str):
    try:
        from pyspark.sql import SparkSession
    except ImportError as error:
        raise RuntimeError("Install the Spark extra with `pip install -e .[spark]`.") from error
    # The Lending Club CSV needs multiLine parsing, which reads the file as one
    # unsplittable task; the 1g local-mode default heap runs out of memory on it.
    driver_memory = os.environ.get("SPARK_DRIVER_MEMORY", "6g")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .getOrCreate()
    )


def process_credit_data(input_path: str | Path, output_path: str | Path) -> None:
    source = require_real_dataset(input_path, "Lending Club")
    spark = _spark_session("credit-risk-processing")
    try:
        loans = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("escape", '"')
            .csv(str(source))
        )
        required = {
            "loan_status", "issue_d", "earliest_cr_line", "loan_amnt", "installment",
            "annual_inc", "dti", "fico_range_low", "fico_range_high",
        }
        missing = required.difference(loans.columns)
        if missing:
            raise ValueError(f"Lending Club input is missing columns: {sorted(missing)}")
        processed = build_credit_features(loans)
        processed.write.mode("overwrite").parquet(str(output_path))
    finally:
        spark.stop()


def process_fraud_data(input_path: str | Path, output_path: str | Path) -> None:
    source = require_real_dataset(input_path, "IEEE-CIS fraud")
    spark = _spark_session("fraud-processing")
    try:
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
        processed = build_fraud_features(transactions)
        processed.write.mode("overwrite").parquet(str(output_path))
    finally:
        spark.stop()


def _try_cast(column: str, data_type: str):
    from pyspark.sql import functions as F

    # Spark 4 runs in ANSI mode, where a plain cast of a malformed value raises.
    return F.expr(f"try_cast(`{column}` AS {data_type})")


def _leading_int(column: str):
    from pyspark.sql import functions as F

    # " 36 months" -> 36, "10+ years" -> 10; empty matches become null.
    return F.expr(f"try_cast(nullif(regexp_extract(`{column}`, '(\\\\d+)', 1), '') AS int)")


def _month_year(column: str):
    from pyspark.sql import functions as F

    return F.to_date(F.expr(f"try_to_timestamp(`{column}`, 'MMM-yyyy')"))


def build_credit_features(loans):
    from pyspark.sql import functions as F

    kept = [name for name in CREDIT_ORIGINATION_COLUMNS if name in loans.columns]
    resolved = loans.select(*kept).where(F.col("loan_status").isin(*DEFAULT_STATUSES, *PAID_STATUSES))

    for name in (
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "fico_range_low",
        "fico_range_high", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
        "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    ):
        if name in resolved.columns:
            resolved = resolved.withColumn(name, _try_cast(name, "double"))

    monthly_income = F.when(F.col("annual_inc") > 0, F.col("annual_inc") / 12)
    history_months = F.months_between(F.col("issue_date"), F.col("earliest_cr_line_date"))
    return (
        resolved.withColumn("default_flag", F.col("loan_status").isin(*DEFAULT_STATUSES).cast("int"))
        .withColumn("issue_date", _month_year("issue_d"))
        .withColumn("earliest_cr_line_date", _month_year("earliest_cr_line"))
        .withColumn("credit_history_months", F.floor(history_months))
        .withColumn(
            "credit_history_bucket",
            F.when(F.col("credit_history_months").isNull(), None)
            .when(F.col("credit_history_months") < 36, "<3y")
            .when(F.col("credit_history_months") < 60, "3-5y")
            .when(F.col("credit_history_months") < 120, "5-10y")
            .when(F.col("credit_history_months") < 240, "10-20y")
            .otherwise("20y+"),
        )
        .withColumn("term_months", _leading_int("term"))
        .withColumn(
            "emp_length_years",
            F.when(F.col("emp_length").startswith("<"), 0).otherwise(_leading_int("emp_length")),
        )
        # Lending Club encodes unavailable DTI as negative values.
        .withColumn("dti", F.when(F.col("dti") >= 0, F.col("dti")))
        .withColumn("fico_mid", (F.col("fico_range_low") + F.col("fico_range_high")) / 2)
        .withColumn("income_missing", F.col("annual_inc").isNull().cast("int"))
        .withColumn("loan_to_income", F.col("loan_amnt") / F.when(F.col("annual_inc") > 0, F.col("annual_inc")))
        .withColumn("installment_to_income", F.col("installment") / monthly_income)
        .dropna(subset=["fico_mid", "issue_date"])
    )


def build_fraud_features(transactions):
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    # card1 alone is shared by many cardholders; combining it with billing address
    # and purchaser email domain gives a closer proxy for a single card/account.
    key_parts = [
        F.coalesce(F.col(name).cast("string"), F.lit("NA"))
        for name in FRAUD_CARD_KEY_COLUMNS
        if name in transactions.columns
    ]
    card_key = F.concat_ws("|", *key_parts) if key_parts else F.col("TransactionID").cast("string")

    processed = (
        transactions.withColumn("card_key", card_key)
        .withColumn("transaction_amount", F.col("TransactionAmt").cast("double"))
        .withColumn("fraud_flag", F.col("isFraud").cast("int"))
    )
    ordered = Window.partitionBy("card_key").orderBy("TransactionDT")
    for label, seconds in FRAUD_VELOCITY_WINDOWS.items():
        # rangeBetween(-N, 0) is backward-looking and includes the current row,
        # so features only use information available at transaction time.
        window = ordered.rangeBetween(-seconds, 0)
        processed = processed.withColumn(f"tx_count_{label}", F.count("TransactionID").over(window))
        processed = processed.withColumn(f"tx_amount_{label}", F.sum("transaction_amount").over(window))
    return (
        processed.withColumn(
            "amount_to_7d_mean",
            F.col("transaction_amount") / (F.col("tx_amount_7d") / F.col("tx_count_7d")),
        )
        .withColumn(
            "seconds_since_prev_tx",
            F.col("TransactionDT") - F.lag("TransactionDT").over(ordered),
        )
    )