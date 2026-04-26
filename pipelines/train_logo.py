import argparse

import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from core.features import LabeledDatasetFeatures
from pipelines.artifacts import logo_artifacts, logo_suffix
from pipelines.config import LogoPipelineConfig, load_config
from pipelines.reporting import print_summary, save_summary, summarize_run_metrics
from steps.encode_features import feature_transformation
from steps.feature_engineering import engineer_features
from steps.load_data import load_data
from steps.preprocess_data import prepare_eShepherd_data
from steps.train_logo_models import train_models_xgboost_logo

DEFAULT_CONFIG_PATH = "./configs/pipelines/train_logo.yaml"


def train_logo(config: LogoPipelineConfig) -> None:
    suffix = logo_suffix(config.experiment_no, config.data_path, config.use_markov)
    artifacts = logo_artifacts(config.feature_engineering, config.cow_normalize_imu, suffix)
    artifacts.ensure_root()


    df = load_data(config.data_path, sheet_name="", delimiter=",", decimal=".")
    df = prepare_eShepherd_data(
        df,
        experiment_number=config.experiment_no,
        missing_strategy=config.missing_strategy,
        remove_outliers=config.remove_outliers,
        feature_engineering=config.feature_engineering,
        is_lstm=False,
    )

    print(f"Features after preparation: {df.columns}")

    if config.feature_engineering:
        df = engineer_features(df, drop_original_features=config.drop_original_features)
        print(f"Features after engineering: {df.columns}")

    logo = LeaveOneGroupOut()
    groups = df[LabeledDatasetFeatures.ANIMAL_ID.feature_name]

    logo_results = {}
    predictions_and_labels = []

    for train_idx, test_idx in logo.split(df, groups=groups):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_df, encoders, _ = feature_transformation(
                train_df,
                feature_engineering=config.feature_engineering,
                cow_normalize_imu=config.cow_normalize_imu
            )

        test_df, _, _ = feature_transformation(
                test_df,
                encoders=encoders,
                feature_engineering=config.feature_engineering,
                cow_normalize_imu=config.cow_normalize_imu
            )

        left_out_id = test_df[LabeledDatasetFeatures.ANIMAL_ID.feature_name].iloc[0]
        y_pred_raw, y_test = train_models_xgboost_logo(
            train_df,
            test_df,
            base_dir=str(artifacts.root),
            use_markov=config.use_markov,
            group_id=left_out_id,
            results_dict=logo_results,
            xgb_params=config.best_params,
            suffix=suffix
        )
    
        predictions_and_labels.append(pd.DataFrame({
            "cow_id": left_out_id,
            "y_true": y_test.astype(int),
            "y_pred": y_pred_raw.astype(float),
        }))

    pd.concat(predictions_and_labels, ignore_index=True).to_parquet(artifacts.predictions_path, index=False)

    # Aggregate metrics across LOGO folds (one left-out cow per fold).
    fold_metrics = []
    numeric_keys = [
        "macro_f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_lying",
        "f1_standing",
        "macro_f1_raw",
        "balanced_accuracy_raw",
        "precision_raw",
        "recall_raw",
        "f1_lying_raw",
        "f1_standing_raw",
    ]

    for cow_id, metrics in logo_results.items():
        model_metrics = {}
        for k in numeric_keys:
            if k in metrics and metrics[k] is not None:
                model_metrics[k] = float(metrics[k])
        if model_metrics:
            fold_metrics.append({"xgboost_logo": model_metrics})

    summary = summarize_run_metrics(fold_metrics)
    title = "LOGO SUMMARY (aggregate across left-out cows)"
    print_summary(title, summary)
    save_summary(
        str(artifacts.aggregate_metrics_path),
        title=title,
        summary=summary,
        run_metrics=fold_metrics,
        extra={
            "n_logo_folds": int(len(fold_metrics)),
            "use_markov": bool(config.use_markov),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LOGO pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(LogoPipelineConfig, args.config)
    train_logo(config)