from pathlib import Path

import pytest

from credit_risk_fraud.config import project_paths
from credit_risk_fraud.validation import require_real_dataset


def test_project_paths_are_stable(tmp_path: Path):
    paths = project_paths(tmp_path)
    paths.create_output_dirs()
    assert paths.processed == tmp_path / "data" / "processed"
    assert paths.models.is_dir()
    assert paths.reports.is_dir()


def test_missing_dataset_explains_real_data_requirement(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="synthetic data is not supported"):
        require_real_dataset(tmp_path / "missing.csv", "test")