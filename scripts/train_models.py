import argparse
import json

import numpy as np
import pandas as pd

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.modeling import (
    evaluate_classifier,
    select_threshold,
    time_ordered_split,
    train_logistic_model,
)

CREDIT_NUMERIC = [
    "loan_amnt", "term_months", "annual_inc", "income_missing", "dti", "fico_mid",
    "emp_length_years", "credit_history_months", "loan_to_income", "installment_to_income",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
]
# grade, sub_grade and int_rate are outputs of Lending Club's own risk model; using them
# would make this model a proxy for theirs, so they stay in the warehouse for BI only.
CREDIT_CATEGORICAL = [
    "purpose", "home_ownership", "verification_status", "application_type", "credit_history_bucket",
]
FRAUD_EXCLUDE = {"TransactionID", "TransactionDT", "isFraud", "fraud_flag"}


def _impute_median(train_df, other_dfs, columns):
    medians = train_df[columns].median()
    return [df[columns].fillna(medians) for df in (train_df, *other_dfs)]


def _one_hot(train_df, other_dfs, columns):
    """Encode categoricals using the training split's levels so unseen levels map to all zeros."""
    train_encoded = pd.get_dummies(train_df[columns], dtype=float)
    return [train_encoded] + [
        pd.get_dummies(df[columns], dtype=float).reindex(columns=train_encoded.columns, fill_value=0.0)
        for df in other_dfs
    ]


def _split_summary(splits, time_column, label_column):
    return {
        name: {
            "rows": len(df),
            "start": str(df[time_column].min()),
            "end": str(df[time_column].max()),
            "positive_rate": float(df[label_column].mean()),
        }
        for name, df in splits.items()
    }


def train_credit_model(paths, minimum_recall: float, max_issue_date: str) -> dict:
    data = pd.read_parquet(paths.processed / "credit")
    data["issue_date"] = pd.to_datetime(data["issue_date"])
    # Recent vintages have mostly not reached a final outcome; keeping only resolved loans
    # from them would over-represent early payoffs and fast defaults.
    data = data[data["issue_date"] < pd.Timestamp(max_issue_date)]
    train_df, val_df, test_df = time_ordered_split(data, "issue_date")

    numeric = _impute_median(train_df, (val_df, test_df), CREDIT_NUMERIC)
    categorical = _one_hot(train_df, (val_df, test_df), CREDIT_CATEGORICAL)
    x_train, x_val, x_test = (pd.concat(parts, axis=1) for parts in zip(numeric, categorical))
    y_train, y_val, y_test = (df["default_flag"] for df in (train_df, val_df, test_df))

    model = train_logistic_model(x_train, y_train)
    threshold = select_threshold(model, x_val, y_val, minimum_recall)
    report = evaluate_classifier(model, x_test, y_test, threshold)
    coefficients = dict(
        zip(x_train.columns, model.named_steps["logisticregression"].coef_[0].astype(float))
    )

    return {
        "model": "credit_default",
        "features": list(x_train.columns),
        "max_issue_date": max_issue_date,
        "splits": _split_summary(
            {"train": train_df, "validation": val_df, "test": test_df}, "issue_date", "default_flag"
        ),
        "report": vars(report),
        "standardized_coefficients": dict(
            sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
        ),
    }


def train_fraud_model(paths, minimum_recall: float) -> dict:
    data = pd.read_parquet(paths.processed / "fraud")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in FRAUD_EXCLUDE]
    train_df, val_df, test_df = time_ordered_split(data, "TransactionDT")

    x_train, x_val, x_test = _impute_median(train_df, (val_df, test_df), feature_cols)
    y_train, y_val, y_test = (df["fraud_flag"] for df in (train_df, val_df, test_df))

    model = train_logistic_model(x_train, y_train)
    threshold = select_threshold(model, x_val, y_val, minimum_recall)
    report = evaluate_classifier(model, x_test, y_test, threshold)

    return {
        "model": "fraud_detection",
        "features": feature_cols,
        "splits": _split_summary(
            {"train": train_df, "validation": val_df, "test": test_df}, "TransactionDT", "fraud_flag"
        ),
        "report": vars(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit and fraud baseline models.")
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    parser.add_argument("--credit", action="store_true", help="Train the credit default model.")
    parser.add_argument("--fraud", action="store_true", help="Train the fraud detection model.")
    parser.add_argument("--minimum-recall", type=float, default=0.5, help="Recall floor for threshold selection.")
    parser.add_argument(
        "--credit-max-issue-date",
        default="2016-01-01",
        help="Exclude loans issued on or after this date (vintages without mature outcomes).",
    )
    args = parser.parse_args()
    if not args.credit and not args.fraud:
        parser.error("select at least one of --credit or --fraud")

    paths = project_paths(args.root)
    paths.create_output_dirs()

    if args.credit:
        result = train_credit_model(paths, args.minimum_recall, args.credit_max_issue_date)
        (paths.reports / "credit_report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))

    if args.fraud:
        result = train_fraud_model(paths, args.minimum_recall)
        (paths.reports / "fraud_report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
