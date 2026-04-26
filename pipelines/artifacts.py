from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def test_dir(self) -> Path:
        return self.root / "test"

    @property
    def preds_dir(self) -> Path:
        return self.root / "preds"

    @property
    def aggregate_metrics_path(self) -> Path:
        return self.root / "aggregate_metrics.json"

    @property
    def predictions_path(self) -> Path:
        return self.root / "predictions.parquet"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_train_test_dirs(self) -> None:
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def ensure_preds_dir(self) -> None:
        self.preds_dir.mkdir(parents=True, exist_ok=True)


def _dataset_stem(data_path: str) -> str:
    return Path(data_path).stem


def classifier_suffix(experiment_no: int, data_path: str, use_markov: bool, split_strategy: str) -> str:
    suffix = f"exp_{experiment_no}_{_dataset_stem(data_path)}"
    if use_markov:
        suffix = f"{suffix}_MARKOV"
    return f"{suffix}_{split_strategy.upper()}"


def classifier_artifacts(suffix: str) -> ArtifactPaths:
    return ArtifactPaths(Path("./data/final_results") / suffix)


def logo_suffix(experiment_no: int, data_path: str, use_markov: bool) -> str:
    suffix = f"exp_{experiment_no}_{_dataset_stem(data_path)}"
    if use_markov:
        suffix = f"{suffix}_MARKOV"
    return suffix


def logo_artifacts(feature_engineering: bool, cow_normalize_imu: bool, suffix: str) -> ArtifactPaths:
    if feature_engineering:
        root = Path("./data/final_results/WITH_FEATURE_ENGINEERING")
    else:
        root = Path("./data/final_results/NO_FEATURE_ENGINEERING")

    root = root / ("logo_cownorm" if cow_normalize_imu else "logo") / suffix
    return ArtifactPaths(root)


def lstm_suffix(experiment_no: int, data_path: str) -> str:
    return f"exp_{experiment_no}_{_dataset_stem(data_path)}"


def lstm_artifacts(suffix: str) -> ArtifactPaths:
    return ArtifactPaths(Path("./data/lstm_results") / suffix)


def train_on_all_suffix(experiment_no: int, test_data_path: str, test_cow_selection_strategy: str) -> str:
    return f"exp_{experiment_no}_{_dataset_stem(test_data_path)}_{test_cow_selection_strategy}"


def train_on_all_artifacts(base_out_dir: str, suffix: str) -> ArtifactPaths:
    return ArtifactPaths(Path(base_out_dir) / suffix)