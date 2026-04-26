import json
import os

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Inner validation split on TRAIN cows only (no peeking at left-out cow)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

from core.features import LabeledDatasetFeatures
from core.markov import (
    build_session_ids,
    fit_hmm_params_with_val_tuning,
    hmm_smooth_per_session_gapaware,
)


def train_models_xgboost_logo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_id: str | int,
    base_dir: str,
    label_col: str = None,
    animal_id_col: str = None,
    save_model: bool = True,
    results_dict: dict = None,
    model_suffix: str = "",
    best_model_state: dict = None,
    xgb_params: dict = None,
    suffix: str = "",
    use_markov: bool = True,
    markov_gap_minutes: int = 5,
    markov_seed: int = 42,
):
    """
    Train and evaluate XGBoost in LOGO fashion, using categorical encoding and no normalization/one-hot.
    Optionally applies HMM/Viterbi smoothing on top of predicted probabilities (fold-local, train-only).
    Results are stored in results_dict keyed by group_id.
    """
    if label_col is None:
        label_col = LabeledDatasetFeatures.LABEL.feature_name
    if animal_id_col is None:
        animal_id_col = LabeledDatasetFeatures.ANIMAL_ID.feature_name

    time_col = LabeledDatasetFeatures.DATETIME.feature_name

    # Prepare train/test sets
    X_train = train_df.drop(columns=[label_col, animal_id_col, time_col], errors="ignore")
    X_test = test_df.drop(columns=[label_col, animal_id_col, time_col], errors="ignore")
    y_train = train_df[label_col]
    y_test = test_df[label_col]

    # Identify categorical columns (object or category dtype)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    # XGBoost with categorical support
    if xgb_params is not None:
        xgb_params = xgb_params.copy()
        xgb_params["enable_categorical"] = True
        model = XGBClassifier(**xgb_params)
    else:
        model = XGBClassifier(enable_categorical=True, eval_metric="logloss", random_state=42)

    # --- Train final model on all training cows (LOGO fold training set) ---
    model.fit(X_train, y_train)

    # --- Raw predictions ---
    y_pred_raw = model.predict_proba(X_test)[:, 1]

    # --- Optionally: HMM smoothing on top of probs ---
    y_pred = (y_pred_raw >= 0.5).astype(int)  # default
    hmm_params = None

    if use_markov:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model has no predict_proba(); cannot apply HMM smoothing.")

        # Build sessions (per cow, split by gaps) for alignment with the SAME rows used for probs
        train_sessions = build_session_ids(train_df, animal_id_col, time_col, max_gap_minutes=markov_gap_minutes)
        test_sessions = build_session_ids(test_df, animal_id_col, time_col, max_gap_minutes=markov_gap_minutes)
        train_ts = pd.to_datetime(train_df[time_col])
        test_ts = pd.to_datetime(test_df[time_col])

        groups = train_df[animal_id_col].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=markov_seed)
        (fit_idx, val_idx), = gss.split(X_train, y_train, groups=groups)

        # Fit a separate model on inner-fit cows to generate val_probs on inner-val cows
        model_for_hmm = clone(model)
        model_for_hmm.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

        val_probs = model_for_hmm.predict_proba(X_train.iloc[val_idx])[:, 1]
        y_val = y_train.iloc[val_idx].to_numpy().astype(int)
        val_sessions = train_sessions.iloc[val_idx].to_numpy()
        val_ts = train_ts.iloc[val_idx].to_numpy()

        # Fit/tune HMM params using ONLY training cows (via inner val)
        hmm_params = fit_hmm_params_with_val_tuning(
            train_labels=y_train.to_numpy().astype(int),
            val_probs=val_probs,
            val_labels=y_val,
            val_sessions=val_sessions,
            val_timestamps=val_ts,
        )

        # Apply smoothing on left-out cow
        test_probs = model.predict_proba(X_test)[:, 1]
        y_pred_markov = hmm_smooth_per_session_gapaware(
            probs=test_probs,
            sessions=test_sessions.to_numpy(),
            timestamps=test_ts.to_numpy(),
            params=hmm_params,
        )

    # Metrics (RAW)
    precision_raw = round(precision_score(y_test, y_pred, average="macro"), 3)
    recall_raw = round(recall_score(y_test, y_pred, average="macro"), 3)
    f1_raw = round(f1_score(y_test, y_pred, average="macro"), 3)
    balanced_acc_raw = balanced_accuracy_score(y_test, y_pred)
    report_raw = classification_report(y_test, y_pred, output_dict=True)
    class_0_raw = report_raw.get("0", None)
    class_1_raw = report_raw.get("1", None)

    # Metrics (SMOOTHED) – equals raw if use_markov=False
    precision = round(precision_score(y_test, y_pred_markov if use_markov else y_pred, average="macro"), 3)
    recall = round(recall_score(y_test, y_pred_markov if use_markov else y_pred, average="macro"), 3)
    f1 = round(f1_score(y_test, y_pred_markov if use_markov else y_pred, average="macro"), 3)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_markov if use_markov else y_pred)
    report = classification_report(y_test, y_pred_markov if use_markov else y_pred, output_dict=True)
    class_0_metrics = report.get("0", None)
    class_1_metrics = report.get("1", None)

    # Save only the best model (based on SMOOTHED macro F1 if smoothing is enabled)
    if save_model:
        os.makedirs(base_dir, exist_ok=True)
        model_path = os.path.join(base_dir, "xgboost_logo_best.joblib")
        if best_model_state is not None:
            if (best_model_state.get("best_f1") is None) or (f1 > best_model_state["best_f1"]):
                joblib.dump(model, model_path)
                best_model_state["best_f1"] = f1
                best_model_state["best_group_id"] = group_id
        else:
            joblib.dump(model, model_path)

    # Store results
    if results_dict is not None:
        results_dict[str(group_id)] = {
            # smoothed (main)
            "macro_f1": f1,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1_lying": class_0_metrics["f1-score"] if class_0_metrics else None,
            "f1_standing": class_1_metrics["f1-score"] if class_1_metrics else None,
            "support_lying": class_0_metrics["support"] if class_0_metrics else None,
            "support_standing": class_1_metrics["support"] if class_1_metrics else None,

            # raw (for comparison)
            "macro_f1_raw": f1_raw,
            "balanced_accuracy_raw": balanced_acc_raw,
            "precision_raw": precision_raw,
            "recall_raw": recall_raw,
            "f1_lying_raw": class_0_raw["f1-score"] if class_0_raw else None,
            "f1_standing_raw": class_1_raw["f1-score"] if class_1_raw else None,

            # optional debug info
            "use_markov": use_markov,
        }
        results_path = os.path.join(base_dir, "results.json")
        save_logo_results_json(results_path, results_dict)
    
    return y_pred_raw.astype(float), y_test.astype(int)

def save_logo_results_json(json_path, results_dict):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Directly dump the accumulated LOGO results
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
