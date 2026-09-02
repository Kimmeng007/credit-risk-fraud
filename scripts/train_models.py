import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.modeling import evaluate_classifier, train_logistic_model

CREDIT_FEATURES = ["annual_inc", "dti", "fico_mid", "income_missing"]
FRAUD_EXCLUDE = {"TransactionID", "TransactionDT", "isFraud", "fraud_flag"}


def _impute_median(train_df, test_df, columns):
    medians = train_df[columns].median()
    return train_df[columns].fillna(medians), test_df[columns].fillna(medians)


def train_credit_model(paths, minimum_recall: float) -> dict:
    data = pd.read_parquet(paths.processed / "credit")
    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["default_flag"]
    )
    x_train, x_test = _impute_median(train_df, test_df, CREDIT_FEATURES)
    y_train, y_test = train_df["default_flag"], test_df["default_flag"]

    model = train_logistic_model(x_train, y_train)
    report = evaluate_classifier(model, x_test, y_test, minimum_recall)
    coefficients = {
        name: float(value)
        for name, value in zip(CREDIT_FEATURES, model.named_steps["logisticregression"].coef_[0])
    }

    return {
        "model": "credit_default",
        "features": CREDIT_FEATURES,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "report": vars(report),
        "standardized_coefficients": coefficients,
    }


def train_fraud_model(paths, minimum_recall: float) -> dict:
    data = pd.read_parquet(paths.processed / "fraud")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in FRAUD_EXCLUDE]

    train_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["fraud_flag"]
    )
    x_train, x_test = _impute_median(train_df, test_df, feature_cols)
    y_train, y_test = train_df["fraud_flag"], test_df["fraud_flag"]

    model = train_logistic_model(x_train, y_train)
    report = evaluate_classifier(model, x_test, y_test, minimum_recall)

    return {
        "model": "fraud_detection",
        "features": feature_cols,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "report": vars(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit and fraud baseline models.")
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    parser.add_argument("--credit", action="store_true", help="Train the credit default model.")
    parser.add_argument("--fraud", action="store_true", help="Train the fraud detection model.")
    parser.add_argument("--minimum-recall", type=float, default=0.5, help="Recall floor for threshold selection.")
    args = parser.parse_args()
    if not args.credit and not args.fraud:
        parser.error("select at least one of --credit or --fraud")

    paths = project_paths(args.root)
    paths.create_output_dirs()

    if args.credit:
        result = train_credit_model(paths, args.minimum_recall)
        (paths.reports / "credit_report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))

    if args.fraud:
        result = train_fraud_model(paths, args.minimum_recall)
        (paths.reports / "fraud_report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "features"}, indent=2))


if __name__ == "__main__":
    main()
