import argparse
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.modeling import (
    calibrate_model,
    calibration_table,
    evaluate_classifier,
    select_threshold,
    shap_importance,
    time_ordered_split,
    train_lightgbm_model,
    train_logistic_model,
)
from credit_risk_fraud.monitoring import population_stability_index, psi_status

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # imported after selecting the non-interactive backend

CREDIT_NUMERIC = [
    "loan_amnt", "term_months", "annual_inc", "income_missing", "dti", "fico_mid",
    "emp_length_years", "credit_history_months", "loan_to_income", "installment_to_income",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
]
# grade, sub_grade and int_rate are outputs of Lending Club's own risk model; using them
# would make this model a proxy for theirs, so they stay in the warehouse for BI only.
# addr_state is excluded too: geography can act as a proxy for protected characteristics
# in lending decisions, so it is analyzed statistically but not used to score applicants.
CREDIT_CATEGORICAL = [
    "purpose", "home_ownership", "verification_status", "application_type", "credit_history_bucket",
]
FRAUD_EXCLUDE = {"TransactionID", "TransactionDT", "isFraud", "fraud_flag", "card_key"}


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


def _as_categories(train_df, other_dfs, numeric, categorical):
    """Numeric columns as-is (LightGBM handles NaN) plus categoricals fixed to training levels."""
    frames = [df[numeric + categorical].copy() for df in (train_df, *other_dfs)]
    for name in categorical:
        levels = pd.Categorical(train_df[name].dropna().astype(str)).categories
        for frame in frames:
            values = frame[name].astype("string")
            # Levels unseen in training become missing, which LightGBM handles natively.
            frame[name] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
    return frames


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


def _fit_and_report(name, x, y, minimum_recall):
    """Train the logistic baseline and LightGBM, calibrate LightGBM, and evaluate all on test."""
    (lr_train, lr_val, lr_test), (gb_train, gb_val, gb_test) = x
    y_train, y_val, y_test = y

    baseline = train_logistic_model(lr_train, y_train)
    baseline_report = evaluate_classifier(
        baseline, lr_test, y_test, select_threshold(baseline, lr_val, y_val, minimum_recall)
    )

    booster = train_lightgbm_model(gb_train, y_train, gb_val, y_val)
    booster_report = evaluate_classifier(
        booster, gb_test, y_test, select_threshold(booster, gb_val, y_val, minimum_recall)
    )
    calibrated = calibrate_model(booster, gb_val, y_val)
    calibrated_report = evaluate_classifier(
        calibrated, gb_test, y_test, select_threshold(calibrated, gb_val, y_val, minimum_recall)
    )

    train_scores = calibrated.predict_proba(gb_train)[:, 1]
    test_scores = calibrated.predict_proba(gb_test)[:, 1]
    score_psi = population_stability_index(train_scores, test_scores)

    return baseline, calibrated, booster, {
        "models": {
            "logistic_regression": vars(baseline_report),
            "lightgbm": {**vars(booster_report), "best_iteration": int(booster.best_iteration_)},
            "lightgbm_calibrated": vars(calibrated_report),
        },
        "calibration_test": calibration_table(y_test, test_scores),
        "score_psi_train_vs_test": {"psi": score_psi, "status": psi_status(score_psi)},
    }


def _save_artifacts(paths, name, calibrated, booster, gb_test, features, categorical):
    joblib.dump(
        {"model": calibrated, "features": features, "categorical": categorical},
        paths.models / f"{name}_lightgbm.joblib",
    )
    importance = shap_importance(booster, gb_test)
    top = list(importance.items())[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature for feature, _ in top], [value for _, value in top], color="#3b6ea5")
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(f"{name}: top features by SHAP importance (test sample)")
    fig.tight_layout()
    fig.savefig(paths.reports / f"{name}_shap_importance.png", dpi=120)
    plt.close(fig)
    return importance


def train_credit_model(paths, minimum_recall: float, max_issue_date: str) -> dict:
    data = pd.read_parquet(paths.processed / "credit")
    data["issue_date"] = pd.to_datetime(data["issue_date"])
    # Recent vintages have mostly not reached a final outcome; keeping only resolved loans
    # from them would over-represent early payoffs and fast defaults.
    data = data[data["issue_date"] < pd.Timestamp(max_issue_date)]
    train_df, val_df, test_df = time_ordered_split(data, "issue_date")
    others = (val_df, test_df)

    numeric = _impute_median(train_df, others, CREDIT_NUMERIC)
    one_hot = _one_hot(train_df, others, CREDIT_CATEGORICAL)
    lr_frames = [pd.concat(parts, axis=1) for parts in zip(numeric, one_hot)]
    gb_frames = _as_categories(train_df, others, CREDIT_NUMERIC, CREDIT_CATEGORICAL)
    labels = [df["default_flag"] for df in (train_df, val_df, test_df)]

    baseline, calibrated, booster, results = _fit_and_report(
        "credit", (lr_frames, gb_frames), labels, minimum_recall
    )
    importance = _save_artifacts(
        paths, "credit", calibrated, booster, gb_frames[2], CREDIT_NUMERIC + CREDIT_CATEGORICAL,
        CREDIT_CATEGORICAL,
    )
    coefficients = dict(
        zip(lr_frames[0].columns, baseline.named_steps["logisticregression"].coef_[0].astype(float))
    )
    return {
        "model": "credit_default",
        "max_issue_date": max_issue_date,
        "splits": _split_summary(
            {"train": train_df, "validation": val_df, "test": test_df}, "issue_date", "default_flag"
        ),
        **results,
        "shap_importance": importance,
        "logistic_standardized_coefficients": dict(
            sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)
        ),
    }


def train_fraud_model(paths, minimum_recall: float) -> dict:
    data = pd.read_parquet(paths.processed / "fraud")
    numeric = [c for c in data.select_dtypes(include=[np.number]).columns if c not in FRAUD_EXCLUDE]
    categorical = [
        c for c in data.columns if c not in FRAUD_EXCLUDE and c not in numeric
    ]
    # float32 halves memory for ~400 numeric columns without affecting tree splits.
    data[numeric] = data[numeric].astype("float32")
    train_df, val_df, test_df = time_ordered_split(data, "TransactionDT")
    del data
    others = (val_df, test_df)

    lr_frames = _impute_median(train_df, others, numeric)
    gb_frames = _as_categories(train_df, others, numeric, categorical)
    labels = [df["fraud_flag"] for df in (train_df, val_df, test_df)]
    splits = _split_summary(
        {"train": train_df, "validation": val_df, "test": test_df}, "TransactionDT", "fraud_flag"
    )
    del train_df, val_df, test_df

    _, calibrated, booster, results = _fit_and_report(
        "fraud", (lr_frames, gb_frames), labels, minimum_recall
    )
    importance = _save_artifacts(
        paths, "fraud", calibrated, booster, gb_frames[2], numeric + categorical, categorical
    )
    return {
        "model": "fraud_detection",
        "categorical_features": categorical,
        "splits": splits,
        **results,
        "shap_importance": dict(list(importance.items())[:30]),
    }


def _print_summary(result: dict) -> None:
    print(f"\n== {result['model']}")
    for name, report in result["models"].items():
        print(
            f"  {name:22s} ROC-AUC {report['roc_auc']:.4f}  PR-AUC {report['pr_auc']:.4f}  "
            f"Brier {report['brier']:.4f}  precision {report['precision']:.3f} "
            f"@ recall {report['recall']:.3f} (threshold {report['threshold']:.3f})"
        )
    psi = result["score_psi_train_vs_test"]
    print(f"  score PSI train vs test: {psi['psi']:.4f} ({psi['status']})")
    print("  top SHAP features:", ", ".join(list(result["shap_importance"])[:8]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit and fraud models.")
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
        _print_summary(result)

    if args.fraud:
        result = train_fraud_model(paths, args.minimum_recall)
        (paths.reports / "fraud_report.json").write_text(json.dumps(result, indent=2))
        _print_summary(result)


if __name__ == "__main__":
    main()
