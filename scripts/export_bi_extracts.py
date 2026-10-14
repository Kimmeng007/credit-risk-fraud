import argparse
import json

import pandas as pd

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.warehouse import connect, run_sql_script

BI_VIEWS = (
    "v_bi_credit_segments",
    "v_bi_credit_state",
    "v_bi_fraud_daily",
    "v_bi_fraud_trend",
    "v_bi_fraud_patterns",
)


def export_views(connection, output_dir) -> None:
    with connection.cursor() as cursor:
        for view in BI_VIEWS:
            cursor.execute(f"SELECT * FROM {view}")
            columns = [description[0].lower() for description in cursor.description]
            frame = pd.DataFrame(cursor.fetchall(), columns=columns)
            frame.to_csv(output_dir / f"{view.removeprefix('v_bi_')}.csv", index=False)
            print(f"  {view}: {len(frame):,} rows")


def export_model_reports(reports_dir, output_dir) -> None:
    """Flatten the training reports into tables Power BI can chart directly."""
    metrics, calibration = [], []
    for task in ("credit", "fraud"):
        report = json.loads((reports_dir / f"{task}_report.json").read_text())
        psi = report["score_psi_train_vs_test"]["psi"]
        for model, values in report["models"].items():
            metrics.append({"task": task, "model": model, "score_psi": psi,
                            **{k: v for k, v in values.items() if k != "best_iteration"}})
        calibration += [{"task": task, **row} for row in report["calibration_test"]]
    pd.DataFrame(metrics).to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output_dir / "model_calibration.csv", index=False)
    print(f"  model_metrics: {len(metrics)} rows, model_calibration: {len(calibration)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create BI views in Oracle and export CSV extracts for Power BI.")
    parser.add_argument("--root", default=None, help="Project root; defaults to the package project root.")
    args = parser.parse_args()
    paths = project_paths(args.root)
    output_dir = paths.root / "data" / "bi"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_dir}")
    with connect() as connection:
        run_sql_script(connection, paths.root / "sql" / "05_bi_views.sql")
        export_views(connection, output_dir)
    export_model_reports(paths.reports, output_dir)


if __name__ == "__main__":
    main()
