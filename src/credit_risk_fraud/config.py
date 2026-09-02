from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path

    @property
    def raw_credit(self) -> Path:
        return self.root / "data" / "raw" / "lending-club"

    @property
    def raw_fraud(self) -> Path:
        return self.root / "data" / "raw" / "ieee-cis"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def models(self) -> Path:
        return self.root / "data" / "models"

    @property
    def reports(self) -> Path:
        return self.root / "data" / "reports"

    def create_output_dirs(self) -> None:
        for path in (self.processed, self.models, self.reports):
            path.mkdir(parents=True, exist_ok=True)


def project_paths(root: str | Path | None = None) -> PipelinePaths:
    project_root = Path(root) if root else Path(__file__).resolve().parents[2]
    return PipelinePaths(project_root)