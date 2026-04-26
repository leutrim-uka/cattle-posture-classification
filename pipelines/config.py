from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Type, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class ClassifierPipelineConfig:
    data_path: str = "./data/CWB_2024.csv"
    experiment_no: int = 5
    feature_engineering: bool = False
    use_markov: bool = False
    missing_strategy: str = "keep"
    remove_outliers: bool = False
    drop_original_features: bool = False
    test_size: float = 0.2
    param_grid_path: str = "./configs/parameter_grids.yaml"
    models_to_train: List[str] = None
    use_smote: bool = False
    cv_folds: int = 5
    split_strategy: str = "random"
    grouped_seed_start: int = 42
    grouped_seed_end: int = 52
    random_cv_folds: int = 5
    random_cv_seed: int = 42

    def __post_init__(self) -> None:
        if self.models_to_train is None:
            self.models_to_train = ["xgboost"]


@dataclass
class LogoPipelineConfig:
    data_path: str = "./data/2025_merged_collar_herde_bolus_weather.csv"
    experiment_no: int = 1
    feature_engineering: bool = False
    use_markov: bool = False
    cow_normalize_imu: bool = False
    missing_strategy: str = "keep"
    remove_outliers: bool = False
    drop_original_features: bool = False
    best_params: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.best_params is None:
            self.best_params = {
                "objective": "binary:logistic",
                "colsample_bytree": 0.9,
                "eval_metric": "logloss",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 300,
                "random_state": 42,
                "subsample": 0.9,
            }


@dataclass
class LSTMPipelineConfig:
    data_path: str = "./data/FINAL_merged_collar_herde_weather_bolus.csv"
    experiment_no: int = 4
    feature_engineering: bool = False
    attention: bool = False
    missing_strategy: str = "keep"
    remove_outliers: bool = False
    is_lstm: bool = True
    drop_original_features: bool = False
    test_size: float = 0.2
    val_size: float = 0.1
    seed_start: int = 42
    seed_end: int = 52


@dataclass
class TrainOnAllDataConfig:
    train_data_path: str = "./data/FINAL_merged_collar_herde_weather_bolus.csv"
    test_data_path: str = "./data/2025_merged_collar_herde_bolus_weather.csv"
    experiment_no: int = 8
    test_cow_selection_strategy: str = "new_cows_test"
    feature_engineering: bool = False
    best_params: Dict[str, Any] = None
    missing_strategy: str = "keep"
    remove_outliers: bool = False
    drop_original_features: bool = False
    out_dir: str = "./data/final_results/final_experiment/FINAL/"

    def __post_init__(self) -> None:
        if self.best_params is None:
            self.best_params = {
                "objective": "binary:logistic",
                "colsample_bytree": 0.9,
                "eval_metric": "logloss",
                "learning_rate": 0.1,
                "max_depth": 3,
                "n_estimators": 300,
                "random_state": 42,
                "subsample": 0.9,
            }


def _load_yaml_file(config_path: str | Path) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping/dict: {path}")

    return data


def _dataclass_from_dict(config_cls: Type[T], payload: Dict[str, Any]) -> T:
    allowed = {f.name for f in fields(config_cls)}
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown keys for {config_cls.__name__}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}"
        )
    return config_cls(**payload)


def load_config(config_cls: Type[T], config_path: str | Path) -> T:
    payload = _load_yaml_file(config_path)
    return _dataclass_from_dict(config_cls, payload)
