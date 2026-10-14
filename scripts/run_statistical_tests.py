import argparse
import json

import pandas as pd

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.statistical_tests import bonferroni_alpha, chi_square_test, mann_whitney_test


def credit_tests(paths, max_issue_date: str) -> list:
    loans = pd.read_parquet(
        paths.processed / "credit",
        columns=["default_flag", "issue_date", "purpose", "addr_state", "annual_inc", "term_months",
                 "home_ownership", "fico_mid", "dti", "credit_history_bucket"],
    )
    # Same mature-vintage restriction as model training, for the same censoring reason.
    loans = loans[pd.to_datetime(loans["issue_date"]) < pd.Timestamp(max_issue_date)]
    # Income quintiles: banding turns a skewed continuous variable into comparable groups.
    loans["income_quintile"] = pd.qcut(loans["annual_inc"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    loans["term_months"] = loans["term_months"].astype(str) + " months"
    return [
        chi_square_test(loans, "purpose", "default_flag", min_count=1000),
        chi_square_test(loans, "addr_state", "default_flag", min_count=1000),
        chi_square_test(loans, "income_quintile", "default_flag"),
        chi_square_test(loans, "term_months", "default_flag"),
        chi_square_test(loans, "home_ownership", "default_flag", min_count=1000),
        chi_square_test(loans, "credit_history_bucket", "default_flag"),
        mann_whitney_test(loans, "fico_mid", "default_flag"),
        mann_whitney_test(loans, "dti", "default_flag"),
    ]


def fraud_tests(paths) -> list:
    transactions = pd.read_parquet(
        paths.processed / "fraud",
        columns=["fraud_flag", "ProductCD", "card6", "DeviceType", "transaction_amount",
                 "tx_count_24h", "seconds_since_prev_tx"],
    )
    return [
        chi_square_test(transactions, "ProductCD", "fraud_flag"),
        chi_square_test(transactions, "card6", "fraud_flag", min_count=1000),
        chi_square_test(transactions, "DeviceType", "fraud_flag"),
        mann_whitney_test(transactions, "transaction_amount", "fraud_flag"),
        mann_whitney_test(transactions, "tx_count_24h", "fraud_flag"),
        mann_whitney_test(transactions, "seconds_since_prev_tx", "fraud_flag"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypothesis tests on default and fraud drivers.")
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    parser.add_argument("--credit-max-issue-date", default="2016-01-01",
                        help="Exclude loans issued on or after this date (immature vintages).")
    args = parser.parse_args()
    paths = project_paths(args.root)
    paths.create_output_dirs()

    results = credit_tests(paths, args.credit_max_issue_date) + fraud_tests(paths)
    alpha = bonferroni_alpha(results)
    report = {
        "bonferroni_alpha": alpha,
        "note": "Large samples make nearly every test significant; read effect sizes for practical importance.",
        "tests": [{**r.as_dict(), "significant": r.p_value < alpha} for r in results],
    }
    (paths.reports / "statistical_tests.json").write_text(json.dumps(report, indent=2))

    print(f"Bonferroni-adjusted alpha: {alpha:.4g} ({len(results)} tests)\n")
    print(f"{'outcome':13s} {'driver':22s} {'test':24s} {'p-value':>10s}  {'effect':>22s}  magnitude")
    for r in results:
        effect = f"{r.effect_size_name}={r.effect_size:.3f}"
        print(f"{r.outcome:13s} {r.driver:22s} {r.test:24s} {r.p_value:10.2e}  {effect:>22s}  {r.effect_magnitude}")


if __name__ == "__main__":
    main()
