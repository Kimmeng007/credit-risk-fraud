from pathlib import Path


def require_real_dataset(path: str | Path, dataset_name: str) -> Path:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found at {dataset_path}. "
            "Download the original dataset and place it under data/raw; synthetic data is not supported."
        )
    if dataset_path.is_file() and dataset_path.stat().st_size == 0:
        raise ValueError(f"{dataset_name} dataset is empty: {dataset_path}")
    return dataset_path