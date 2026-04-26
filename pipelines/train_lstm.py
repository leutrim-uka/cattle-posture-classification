import argparse

from pipelines.artifacts import lstm_artifacts, lstm_suffix
from pipelines.config import LSTMPipelineConfig, load_config
from pipelines.reporting import print_summary, save_summary, summarize_run_metrics
from steps.encode_features import feature_transformation
from steps.feature_engineering import engineer_features
from steps.load_data import load_data
from steps.lstm_preprocessing import create_lstm_sequences
from steps.preprocess_data import grouped_train_test_split, prepare_eShepherd_data
from steps.train_lstm import train_lstm_model, train_lstm_with_attention

DEFAULT_CONFIG_PATH = "./configs/pipelines/train_lstm.yaml"


# @pipeline(name="lstm_pipeline")
def train_lstm(config: LSTMPipelineConfig):
    suffix = lstm_suffix(config.experiment_no, config.data_path)
    root_artifacts = lstm_artifacts(suffix)
    root_artifacts.ensure_root()

    df = load_data(config.data_path, sheet_name="", delimiter=",", decimal=".")
    df = prepare_eShepherd_data(
        df,
        experiment_number=config.experiment_no,
        missing_strategy=config.missing_strategy,
        remove_outliers=config.remove_outliers,
        feature_engineering=config.feature_engineering,
        is_lstm=config.is_lstm,
    )

    if config.feature_engineering:
        df = engineer_features(df, drop_original_features=config.drop_original_features)

    run_metrics = []

    for seed in range(config.seed_start, config.seed_end):
        train_df, test_df = grouped_train_test_split(df, seed, config.test_size)
        val_df = grouped_train_test_split(train_df, seed, test_size=config.val_size)[1]
        run_artifacts = lstm_artifacts(f"{suffix}_SEED_{seed}")
        run_artifacts.ensure_root()

        # STEP 6: Encode and embed features
        train_df, encoders, _ = feature_transformation(
            train_df,
        )

        val_df, _, _ = feature_transformation(
            val_df,
            encoders=encoders,
        )

        test_df, _, _ = feature_transformation(
            test_df,
            encoders=encoders,
        )

        X_train, y_train, X_val, y_val, X_test, y_test = create_lstm_sequences(train_df, val_df, test_df)

        if config.attention:
            metrics = train_lstm_with_attention(
                X_train,
                X_test,
                y_train,
                y_test,
                seed=seed,
                output_dir=str(run_artifacts.root),
            )
        else:
            metrics = train_lstm_model(
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                seed=seed,
                output_dir=str(run_artifacts.root),
            )

        run_metrics.append(metrics)

    summary = summarize_run_metrics(run_metrics)
    n_runs = config.seed_end - config.seed_start
    title = f"LSTM SUMMARY ({n_runs} grouped splits)"
    print_summary(title, summary)
    save_summary(
        str(root_artifacts.aggregate_metrics_path),
        title=title,
        summary=summary,
        run_metrics=run_metrics,
        extra={
            "attention": bool(config.attention),
            "seed_start": config.seed_start,
            "seed_end": config.seed_end,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    config = load_config(LSTMPipelineConfig, args.config)
    train_lstm(config)