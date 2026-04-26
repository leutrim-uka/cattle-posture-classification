import datetime
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from hmmlearn import hmm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # SMOTE is from imblearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from core.features import LabeledDatasetFeatures
from core.log import get_logger
from core.markov import (
    build_session_ids,
    fit_hmm_params_with_val_tuning,
    hmm_smooth_per_session_gapaware,
)

logger = get_logger(__name__)

def load_model_params(file_path: str):
    with open(file_path, "r") as f:
        param_grids = yaml.safe_load(f)

    model_mapping = {
        "logreg": LogisticRegression(),
        "rf": RandomForestClassifier(),
        "gb": GradientBoostingClassifier(),
        "svm": SVC(probability=True),
        "nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "dt": DecisionTreeClassifier(),
        "xgboost": XGBClassifier(eval_metric="logloss"),
    }

    return model_mapping, param_grids


def train_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    models_to_train: list,
    param_grid_path: str,
    use_smote: bool,
    cv_folds: int,
    use_markov: bool,
    suffix: str,
    output_dir: str | None = None,
):
    results = {}
    run_metrics = {}

    model_dir = output_dir or f"./data/final_results/{suffix}/"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(model_dir, "test"), exist_ok=True)

    models, param_grids = load_model_params(param_grid_path)

    group_ids = train_df[
        LabeledDatasetFeatures.ANIMAL_ID.feature_name
    ]  # Cow IDs for GroupKFold

    X_train = train_df.drop(
        columns=[
            LabeledDatasetFeatures.ANIMAL_ID.feature_name,
            LabeledDatasetFeatures.LABEL.feature_name,
            LabeledDatasetFeatures.DATETIME.feature_name
        ]
    )
    X_test = test_df.drop(
        columns=[
            LabeledDatasetFeatures.ANIMAL_ID.feature_name,
            LabeledDatasetFeatures.LABEL.feature_name,
            LabeledDatasetFeatures.DATETIME.feature_name
        ]
    )

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Test df shape: {test_df.shape}")

    train_df.to_pickle(os.path.join(model_dir, f"train/X_train_{suffix}.pkl"))
    test_df.to_pickle(os.path.join(model_dir, f"test/X_test_{suffix}.pkl"))

    y_train = train_df[LabeledDatasetFeatures.LABEL.feature_name]
    y_test = test_df[LabeledDatasetFeatures.LABEL.feature_name]

    cv = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")

        model = models[model_name]
        params = param_grids[model_name]

        if "random_state" in model.get_params():
            model.set_params(random_state=42)

        if use_smote:
            steps = []
            if use_smote:
                logger.info("using SMOTE")
                steps.append(("smote", SMOTE(random_state=42)))
            steps.append(("model", models[model_name]))

            pipeline = ImbPipeline(steps)

            grid_search = RandomizedSearchCV(
                pipeline,
                param_grid={f"model__{key}": value for key, value in params.items()},
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                n_iter=10,  # Try 10 random combinations
                random_state=42
            )
        else:
            grid_search = RandomizedSearchCV(
                model,
                params,
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                n_iter=10,  # Try 10 random combinations
                random_state=42
            )

        grid_search.fit(X_train, y_train, groups=group_ids)
        best_model = grid_search.best_estimator_

        # Save each model with seed and model name for uniqueness
        model_path = os.path.join(model_dir, f"{model_name}_seed_{seed}_{suffix}.joblib")
        joblib.dump(best_model, model_path)

        # Predict
        if not use_markov:
            y_pred = best_model.predict(X_test)
        else:
            logger.info("#######################")
            logger.info("Using HMM smoothing")
            logger.info("#######################")

            if not hasattr(best_model, "predict_proba"):
                raise ValueError(f"{model_name} has no predict_proba(), can't apply HMM.")

            # --- Build sessions ---
            aid = LabeledDatasetFeatures.ANIMAL_ID.feature_name
            ts  = LabeledDatasetFeatures.DATETIME.feature_name
            train_sessions = build_session_ids(train_df, aid, ts)
            test_sessions  = build_session_ids(test_df, aid, ts)
            train_ts = pd.to_datetime(train_df[ts])
            test_ts  = pd.to_datetime(test_df[ts])

            # --- Make a small validation split from training cows ---
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
            (fit_idx, val_idx), = gss.split(X_train, y_train, groups=group_ids)

            # Fit a model for VAL probs (consistent with your current setup)
            model_for_hmm = clone(best_model)
            model_for_hmm.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

            val_probs = model_for_hmm.predict_proba(X_train.iloc[val_idx])[:, 1]
            y_val = y_train.iloc[val_idx].to_numpy().astype(int)

            # Sessions/timestamps for VAL subset (same rows as val_probs)
            val_sessions = train_sessions.iloc[val_idx].to_numpy()
            val_ts = train_ts.iloc[val_idx].to_numpy()

            # --- Fit HMM params on TRAIN/VAL only (tune A on VAL) ---
            hmm_params = fit_hmm_params_with_val_tuning(
                train_labels=y_train.to_numpy().astype(int),
                val_probs=val_probs,
                val_labels=y_val,
                val_sessions=val_sessions,
                val_timestamps=val_ts,
            )

            # --- Apply on TEST ---
            test_probs = best_model.predict_proba(X_test)[:, 1]
            y_pred = hmm_smooth_per_session_gapaware(
                probs=test_probs,
                sessions=test_sessions.to_numpy(),
                timestamps=test_ts.to_numpy(),
                params=hmm_params,
            )

        # Store predictions
        preds_dir = os.path.join(model_dir, "preds")
        os.makedirs(preds_dir, exist_ok=True)
        pd.Series(y_pred, index=X_test.index).to_pickle(os.path.join(preds_dir, f"y_pred_{model_name}_seed_{seed}_{suffix}.pkl"))
        pd.Series(y_test).to_pickle(os.path.join(preds_dir, f"y_test_{model_name}_seed_{seed}_{suffix}.pkl"))

        # Metrics
        precision = round(precision_score(y_test, y_pred, average="macro"), 3)
        recall = round(recall_score(y_test, y_pred, average="macro"), 3)
        f1 = round(f1_score(y_test, y_pred, average="macro"), 3)

        report = classification_report(y_test, y_pred, output_dict=True)
        class_0_metrics = report["0"]
        class_1_metrics = report["1"]

        results[model_name] = {
            "Model": best_model,
            "Global Metrics": {
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
            },
            "Liegen Metrics": class_0_metrics,
            "Stehen Metrics": class_1_metrics,
            "X_test": X_test,
        }

        r_g, c_0, c_1 = format_results(results)
    
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"\n✅ Balanced Accuracy: {balanced_acc:.4f}")
        merged_results = merge_results_ordered(r_g, c_0, c_1)
        merged_results["balanced_accuracy"] = balanced_acc
        # Save results JSON in the same model_dir
        results_json_path = os.path.join(model_dir, f"results_{suffix}.json")
        save_results_json(
            results_json_path,
            seed,
            merged_results,
            test_df.animal_id.unique().tolist()
        )

        run_metrics[model_name] = {
            "macro_f1": results[model_name]["Global Metrics"]["F1 Score"],
            "balanced_accuracy": float(balanced_acc),
            "precision": results[model_name]["Global Metrics"]["Precision"],
            "recall": results[model_name]["Global Metrics"]["Recall"],
            # Liegen (class 0)
            "f1_liegen": results[model_name]["Liegen Metrics"]["f1-score"],
            "precision_liegen": results[model_name]["Liegen Metrics"]["precision"],
            "recall_liegen": results[model_name]["Liegen Metrics"]["recall"],
            # Stehen (class 1)
            "f1_stehen": results[model_name]["Stehen Metrics"]["f1-score"],
            "precision_stehen": results[model_name]["Stehen Metrics"]["precision"],
            "recall_stehen": results[model_name]["Stehen Metrics"]["recall"],
        }

    return run_metrics





def format_results(results):
    global_df = pd.DataFrame(
        {model: metrics["Global Metrics"] for model, metrics in results.items()}
    ).T.sort_values(by="F1 Score", ascending=False)
    global_df["Support"] = None

    class_0_df = pd.DataFrame(
        {
            model: {
                "Precision": metrics["Liegen Metrics"]["precision"],
                "Recall": metrics["Liegen Metrics"]["recall"],
                "F1 Score": metrics["Liegen Metrics"]["f1-score"],
                "Support": metrics["Liegen Metrics"]["support"],
            }
            for model, metrics in results.items()
        }
    ).T.sort_values(by="F1 Score", ascending=False)

    class_1_df = pd.DataFrame(
        {
            model: {
                "Precision": metrics["Stehen Metrics"]["precision"],
                "Recall": metrics["Stehen Metrics"]["recall"],
                "F1 Score": metrics["Stehen Metrics"]["f1-score"],
                "Support": metrics["Stehen Metrics"]["support"],
            }
            for model, metrics in results.items()
        }
    ).T.sort_values(by="F1 Score", ascending=False)

    return global_df, class_0_df, class_1_df


def merge_results_ordered(global_df, class_0_df, class_1_df):
    # Add suffixes to identify each group (Global, Class 0, Class 1)
    global_df = global_df.add_suffix(" Global")
    class_0_df = class_0_df.add_suffix(" Liegen")
    class_1_df = class_1_df.add_suffix(" Stehen")

    logger.info(global_df)
    logger.info(class_0_df)
    logger.info(class_1_df)

    # Merge them all into one dataframe
    merged_df = pd.concat([global_df, class_0_df, class_1_df], axis=1)

    logger.info("merged df columns")
    logger.info(global_df.columns)
    # Reorder columns: Precision first, then Recall, F1, etc.
    ordered_columns = []
    for metric in ["Precision", "Recall", "F1 Score", "Support"]:
        ordered_columns.extend(
            [f"{metric} Global", f"{metric} Liegen", f"{metric} Stehen"]
        )

    merged_df = merged_df[ordered_columns]

    return merged_df



def hmm_postprocess(y_proba, y_true):
    # Convert predicted probabilities into pseudo-observations
    obs = y_proba[:, 1].reshape(-1, 1)  # probability of class 1 = standing

    model_hmm = hmm.GaussianHMM(
        n_components=2,
        covariance_type="diag",
        n_iter=100,
        init_params=""  # prevent overwriting
    )
    model_hmm.startprob_ = np.array([0.5, 0.5])
    model_hmm.transmat_ = np.array([
        [0.95, 0.05],
        [0.05, 0.95],
    ])

    # estimated_transmat = estimate_transition_matrix_smoothed(y_true)
    # model_hmm.transmat_ = estimated_transmat

    model_hmm.means_ = np.array([[0.2], [0.8]])  # Lying / Standing mean probs
    model_hmm.covars_ = np.array([[0.05], [0.05]])
    
    # model_hmm.fit(obs)

    smoothed_states = model_hmm.predict(obs)
    return smoothed_states


def estimate_transition_matrix_smoothed(y_true: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    transmat_ = np.zeros((2, 2)) + alpha  # Add alpha to every transition

    for prev, curr in zip(y_true[:-1], y_true[1:]):
        transmat_[prev, curr] += 1

    transmat_ /= transmat_.sum(axis=1, keepdims=True)
    return transmat_


def append_list_to_txt(filepath, items):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "a") as f:
        for item in items:
            f.write(str(item) + "\n")



def save_results_json(json_path, seed, merged_df, test_ids):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    # Load existing file if it exists
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Convert DataFrame to dictionary
    metrics_dict = merged_df.to_dict(orient="index")

    # Store metrics + test IDs under this seed
    all_results[str(seed)] = {
        "IDs": list(map(int, test_ids)),  # ensure serializable
        "Metrics": metrics_dict
    }

    # Save back to file
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)