import argparse

from sklearn.model_selection import StratifiedKFold

from core.features import LabeledDatasetFeatures
from pipelines.artifacts import classifier_artifacts, classifier_suffix
from pipelines.config import ClassifierPipelineConfig, load_config
from pipelines.reporting import print_summary, save_summary, summarize_run_metrics
from steps.encode_features import feature_transformation
from steps.feature_engineering import engineer_features
from steps.load_data import load_data
from steps.preprocess_data import grouped_train_test_split, prepare_eShepherd_data
from steps.train_models import train_models

DEFAULT_CONFIG_PATH = "./configs/pipelines/train_classifier.yaml"


def train_classifier(config: ClassifierPipelineConfig) -> None:
    suffix = classifier_suffix(
        config.experiment_no,
        config.data_path,
        config.use_markov,
        config.split_strategy,
    )
    root_artifacts = classifier_artifacts(suffix)
    root_artifacts.ensure_root()


    df = load_data(config.data_path, sheet_name="", delimiter=",", decimal=".")
    df = prepare_eShepherd_data(
        df,
        experiment_number=config.experiment_no,
        missing_strategy=config.missing_strategy,
        feature_engineering=config.feature_engineering,
        remove_outliers=config.remove_outliers,
        is_lstm=False,
    )

    print(f"Features after preparation: {df.columns}")

    if config.feature_engineering:
        df = engineer_features(df, drop_original_features=config.drop_original_features)
        print(f"Features after engineering: {df.columns}")

    if config.split_strategy not in {"grouped", "random"}:
        raise ValueError("split_strategy must be one of: 'grouped', 'random'")

    def run_single_train_eval(train_df, test_df, run_seed, run_suffix):
        train_df_enc, encoders, _ = feature_transformation(
            train_df,
            feature_engineering=config.feature_engineering,
        )

        test_df_enc, _, _ = feature_transformation(
            test_df,
            encoders=encoders,
            feature_engineering=config.feature_engineering,
        )

        return train_models(
            train_df_enc,
            test_df_enc,
            run_seed,
            models_to_train=config.models_to_train,
            param_grid_path=config.param_grid_path,
            use_smote=config.use_smote,
            cv_folds=config.cv_folds,
            use_markov=config.use_markov,
            suffix=run_suffix,
            output_dir=str(classifier_artifacts(run_suffix).root),
        )

    run_metrics = []

    if config.split_strategy == "grouped":
        for seed in range(config.grouped_seed_start, config.grouped_seed_end):
            train_df, test_df = grouped_train_test_split(df, seed, config.test_size)
            metrics = run_single_train_eval(train_df, test_df, seed, suffix)
            run_metrics.append(metrics)
    else:
        label_col = LabeledDatasetFeatures.LABEL.feature_name
        skf = StratifiedKFold(
            n_splits=config.random_cv_folds,
            shuffle=True,
            random_state=config.random_cv_seed,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(df, df[label_col]),
            start=1,
        ):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # Keep fold runs separate in output artifacts.
            fold_suffix = f"{suffix}_FOLD_{fold_idx}"
            fold_seed = config.random_cv_seed + fold_idx
            metrics = run_single_train_eval(train_df, test_df, fold_seed, fold_suffix)
            run_metrics.append(metrics)

    summary = summarize_run_metrics(run_metrics)
    if config.split_strategy == "random":
        title = f"CV SUMMARY ({config.random_cv_folds}-fold StratifiedKFold, random split)"
    else:
        n_runs = config.grouped_seed_end - config.grouped_seed_start
        title = f"REPEATED HOLDOUT SUMMARY ({n_runs} grouped splits)"

    print_summary(title, summary)

    save_summary(
        str(root_artifacts.aggregate_metrics_path),
        title=title,
        summary=summary,
        run_metrics=run_metrics,
        extra={
            "split_strategy": config.split_strategy,
            "random_cv_folds": config.random_cv_folds,
            "random_cv_seed": config.random_cv_seed,
            "grouped_seed_start": config.grouped_seed_start,
            "grouped_seed_end": config.grouped_seed_end,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(ClassifierPipelineConfig, args.config)
    train_classifier(config)