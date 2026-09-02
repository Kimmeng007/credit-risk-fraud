import argparse

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.spark_pipeline import process_credit_data, process_fraud_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Process original credit and fraud datasets with Spark.")
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    parser.add_argument("--credit", action="store_true", help="Process the Lending Club dataset.")
    parser.add_argument("--fraud", action="store_true", help="Process the IEEE-CIS dataset.")
    args = parser.parse_args()
    if not args.credit and not args.fraud:
        parser.error("select at least one of --credit or --fraud")

    paths = project_paths(args.root)
    paths.create_output_dirs()
    if args.credit:
        process_credit_data(paths.raw_credit, paths.processed / "credit")
    if args.fraud:
        process_fraud_data(paths.raw_fraud, paths.processed / "fraud")


if __name__ == "__main__":
    main()