import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from core.features import LabeledDatasetFeatures
from pipelines.artifacts import train_on_all_artifacts, train_on_all_suffix
from pipelines.config import TrainOnAllDataConfig, load_config
from pipelines.reporting import print_summary, save_summary, summarize_run_metrics
from steps.encode_features import feature_transformation
from steps.feature_engineering import engineer_features
from steps.load_data import load_data
from steps.preprocess_data import prepare_eShepherd_data

DEFAULT_CONFIG_PATH = "./configs/pipelines/train_on_all_data.yaml"


# @pipeline(name="classifier_pipeline")
def train_on_all_data(config: TrainOnAllDataConfig):
    suffix = train_on_all_suffix(
        config.experiment_no,
        config.test_data_path,
        config.test_cow_selection_strategy,
    )
    artifacts = train_on_all_artifacts(config.out_dir, suffix)
    artifacts.ensure_root()

    # Load the data from 2024 as training and 2025 as test set
    train_df = load_data(config.train_data_path, sheet_name="", delimiter=",", decimal=".")
    test_df = load_data(config.test_data_path, sheet_name="", delimiter=",", decimal=".")

    if config.test_cow_selection_strategy == "new_cows_test":
        # Ensure that only new cows from 2025 are used for testing
        # Recurring cows that also appear in 2024 are dropped
        test_df = test_df[~test_df["animal_id"].isin(train_df["animal_id"])]
    elif config.test_cow_selection_strategy == "recurring_cows_test":
        # Ensure that only recurring cows from 2025 are used for testing
        test_df = test_df[test_df["animal_id"].isin(train_df["animal_id"])]
    elif config.test_cow_selection_strategy == "recurring_cows_to_training":
        # extract IDs of cows that appear in both 2024 and 2025 sets
        recurring_cows_ids = set(train_df["animal_id"]).intersection(set(test_df["animal_id"]))
        # take the rows into a separate dataframe and add them to the training set
        recurring_cows_df = test_df[test_df["animal_id"].isin(recurring_cows_ids)]
        train_df = pd.concat([train_df, recurring_cows_df], ignore_index=True)
        train_df.sort_values(by=[LabeledDatasetFeatures.ANIMAL_ID.feature_name, LabeledDatasetFeatures.DATETIME.feature_name], inplace=True)
        # remove recurring cows from test set
        test_df = test_df[~test_df["animal_id"].isin(recurring_cows_ids)]

    # Prepare the data (feature selection, missing value handling, outlier removal, label generation, etc.)
    train_df = prepare_eShepherd_data(
        train_df,
        experiment_number=config.experiment_no,
        missing_strategy=config.missing_strategy,
        feature_engineering=config.feature_engineering,
        remove_outliers=config.remove_outliers,
        is_lstm=False,
    )
    
    test_df = prepare_eShepherd_data(
        test_df,
        experiment_number=config.experiment_no,
        missing_strategy=config.missing_strategy,
        feature_engineering=config.feature_engineering,
        remove_outliers=config.remove_outliers,
        is_lstm=False,
    )
        
    print(f"Features after preparation: {train_df.columns}")

    if config.feature_engineering:
        train_df = engineer_features(train_df, drop_original_features=config.drop_original_features)
        print(f"Features after engineering: {train_df.columns}")

    # Encode and embed features. Return encoders to ensure we use the same encoding for test set.
    train_df, encoders, _ = feature_transformation(
        train_df,
        feature_engineering=config.feature_engineering
    )

    test_df, _, _ = feature_transformation(
        test_df,
        encoders=encoders,
        feature_engineering=config.feature_engineering
    )

    # Drop metadata columns and split into X and y
    X_train = train_df.drop(
        columns=[
            LabeledDatasetFeatures.ANIMAL_ID.feature_name,
            LabeledDatasetFeatures.LABEL.feature_name,
            LabeledDatasetFeatures.DATETIME.feature_name
        ]
    )
    y_train = train_df[LabeledDatasetFeatures.LABEL.feature_name]

    X_test = test_df.drop(
        columns=[
            LabeledDatasetFeatures.ANIMAL_ID.feature_name,
            LabeledDatasetFeatures.LABEL.feature_name,
            LabeledDatasetFeatures.DATETIME.feature_name
        ]
    )

    y_test = test_df[LabeledDatasetFeatures.LABEL.feature_name]

    artifacts.ensure_train_test_dirs()
    train_df.to_pickle(artifacts.train_dir / f"X_train_{suffix}.pkl")
    test_df.to_pickle(artifacts.test_dir / f"X_test_{suffix}.pkl")


    # Note: We keep the animal_id column separately for evaluation purposes, but it is not used as a feature for training.
    animal_ids_test = test_df[LabeledDatasetFeatures.ANIMAL_ID.feature_name]


    # Identify categorical columns (object or category dtype)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # XGBoost with categorical support
    if config.best_params is not None:
        best_params = config.best_params.copy()
        best_params["enable_categorical"] = True
        model = XGBClassifier(**best_params)
    else:
        model = XGBClassifier(enable_categorical=True, eval_metric="logloss", random_state=42)

    # --- Train final model on all training cows (LOGO fold training set) ---
    model.fit(X_train, y_train)

    # --- Raw predictions ---
    y_pred_raw = model.predict_proba(X_test)[:, 1]
    # --- Thresholding at 0.5 to get binary predictions ---
    y_pred = (y_pred_raw >= 0.5).astype(int)

    ####################################
    ####### ROW-LEVEL EVALUATION #######
    ####################################
    # Aggregate (cow-independent) metrics
    precision_raw = round(precision_score(y_test, y_pred, average="macro"), 3)
    recall_raw = round(recall_score(y_test, y_pred, average="macro"), 3)
    f1_raw = round(f1_score(y_test, y_pred, average="macro"), 3)
    balanced_acc_raw = balanced_accuracy_score(y_test, y_pred)


    print("\nAggregate (cow-independent) scores:")
    print(f"Precision: {precision_raw}, Recall: {recall_raw}, F1: {f1_raw}, Balanced Acc: {balanced_acc_raw}")

    #####################################
    ####### PER-COW EVALUATION ##########
    #####################################
    predictions_and_labels = []
    unique_cows = np.unique(animal_ids_test)
    per_cow_scores = []

    for cow_id in unique_cows:
        idx = (animal_ids_test == cow_id)

        y_true_cow = y_test[idx].astype(int)
        y_pred_prob_cow = y_pred_raw[idx].astype(float)
        y_pred_label_cow = y_pred[idx].astype(int)

        # Save predictions and labels for this cow
        predictions_and_labels.append(pd.DataFrame({
            "cow_id": cow_id,
            "y_true": y_true_cow,
            "y_pred": y_pred_prob_cow,
            "y_pred_label": y_pred_label_cow
        }))

        # Supports (used to decide whether per-class metrics are defined)
        support_lying = int((y_true_cow == 0).sum())
        support_standing = int((y_true_cow == 1).sum())

        # Per-cow macro metrics
        precision_cow = precision_score(y_true_cow, y_pred_label_cow, average="macro", zero_division=0)
        recall_cow = recall_score(y_true_cow, y_pred_label_cow, average="macro", zero_division=0)
        f1_cow = f1_score(y_true_cow, y_pred_label_cow, average="macro", zero_division=0)
        balanced_acc_cow = balanced_accuracy_score(y_true_cow, y_pred_label_cow)

        # Per-class F1 for this cow (mask undefined classes as NaN so aggregation ignores them)
        f1_per_class = f1_score(
            y_true_cow,
            y_pred_label_cow,
            average=None,
            labels=[0, 1],
            zero_division=0
        )
        f1_lying_cow = float(f1_per_class[0])
        f1_standing_cow = float(f1_per_class[1])

        if support_lying == 0:
            f1_lying_cow = np.nan
        if support_standing == 0:
            f1_standing_cow = np.nan

        per_cow_scores.append({
            "animal_id": cow_id,
            "precision": round(float(precision_cow), 3),
            "recall": round(float(recall_cow), 3),
            "f1": round(float(f1_cow), 3),  # per-cow macro-F1
            "f1_lying": round(float(f1_lying_cow), 3) if not np.isnan(f1_lying_cow) else np.nan,
            "f1_standing": round(float(f1_standing_cow), 3) if not np.isnan(f1_standing_cow) else np.nan,
            "balanced_acc": float(balanced_acc_cow),
            "support_lying": support_lying,
            "support_standing": support_standing
        })

    per_cow = pd.DataFrame(per_cow_scores)

    print("\nPer-cow scores:")
    for cow_score in per_cow_scores:
        print(cow_score)

    # Optional sanity checks (does not change reporting)
    print("\nSanity checks:")
    print("Cows with no lying samples:", int((per_cow["support_lying"] == 0).sum()))
    print("Cows with no standing samples:", int((per_cow["support_standing"] == 0).sum()))

    #########################################
    ####### COW-AVERAGED AGGREGATION ########
    #########################################
    # This is the SAME aggregation style you used before, but now per-class F1 ignores undefined cows.
    agg_results = per_cow.describe().T[["mean", "std"]]

    # If you want to keep only the metrics you report (nothing more, nothing less):
    agg_results = agg_results.loc[["f1", "f1_lying", "f1_standing", "balanced_acc"], :]

    print("\nCow-averaged scores (mean ± std across cows):")
    print(agg_results)

    ####################################
    ####### SAVE OUTPUTS ###############
    ####################################

    pd.concat(predictions_and_labels, ignore_index=True).to_parquet(
        artifacts.predictions_path,
        index=False
    )

    metrics = {
        "aggregate": {
            "precision": float(precision_raw),
            "recall": float(recall_raw),
            "f1": float(f1_raw),
            "balanced_acc": float(balanced_acc_raw)
        },
        "per_cow": [
            {
                "animal_id": str(row["animal_id"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "f1_lying": None if pd.isna(row["f1_lying"]) else float(row["f1_lying"]),
                "f1_standing": None if pd.isna(row["f1_standing"]) else float(row["f1_standing"]),
                "balanced_acc": float(row["balanced_acc"]),
                "support_lying": int(row["support_lying"]),
                "support_standing": int(row["support_standing"])
            }
            for _, row in per_cow.iterrows()
        ]
    }

    with open(artifacts.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    summary_metrics = {
        "xgboost": {
            "macro_f1": float(f1_raw),
            "balanced_accuracy": float(balanced_acc_raw),
            "precision": float(precision_raw),
            "recall": float(recall_raw),
            "f1_liegen": float(agg_results.loc["f1_lying", "mean"]),
            "f1_stehen": float(agg_results.loc["f1_standing", "mean"]),
            "balanced_acc_cow_mean": float(agg_results.loc["balanced_acc", "mean"]),
        }
    }
    summary = summarize_run_metrics([summary_metrics])
    title = "TRAIN-ON-ALL-DATA SUMMARY (single fixed split)"
    print_summary(title, summary)
    save_summary(
        str(artifacts.aggregate_metrics_path),
        title=title,
        summary=summary,
        run_metrics=[summary_metrics],
        extra={
            "test_cow_selection_strategy": config.test_cow_selection_strategy,
        },
    )

    model_path = artifacts.root / f"xgboost_seed_{suffix}.joblib"
    joblib.dump(model, model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train on full training set and evaluate fixed test set")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline YAML config",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(TrainOnAllDataConfig, args.config)
    train_on_all_data(config)