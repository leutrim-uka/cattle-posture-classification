import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from zenml import step

from core.features import LabeledDatasetFeatures
from core.log import get_logger

logger = get_logger(__name__)


# @step(enable_cache=False)
def create_lstm_sequences(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    window_size: int = 60,
    max_gap_minutes: int = 4,
    min_session_len = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, meta_train = create_sequences(
        train_df, window_size, max_gap_minutes, min_session_len
    )
    X_val, y_val, meta_val = create_sequences(
        val_df, window_size, max_gap_minutes, min_session_len
    )
    X_test, y_test, meta_test = create_sequences(
        test_df, window_size, max_gap_minutes, min_session_len
    )

    meta_test_df = pd.DataFrame(meta_test)
    meta_train_df = pd.DataFrame(meta_train)
    meta_val_df = pd.DataFrame(meta_val)
    meta_train_df.to_csv("./data/lstm_results/X_train_meta.csv", index=False)
    meta_val_df.to_csv("./data/lstm_results/X_val_meta.csv", index=False)
    meta_test_df.to_csv("./data/lstm_results/X_test_meta.csv", index=False)

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(df, window_size, max_gap_minutes, min_session_len):
    X, y, meta = [], [], []
    features = list(
        df.drop(
            columns=[
                LabeledDatasetFeatures.ANIMAL_ID.feature_name,
                LabeledDatasetFeatures.DATETIME.feature_name,
                LabeledDatasetFeatures.LABEL.feature_name,
            ]
        ).columns
    )

    for animal_id, group in df.groupby("animal_id"):
        group = group.sort_values(
            LabeledDatasetFeatures.DATETIME.feature_name
        ).reset_index(drop=True)

        # Compute time differences
        time_diffs = (
            group[LabeledDatasetFeatures.DATETIME.feature_name]
            .diff()
            .dt.total_seconds()
        )
        # New session if gap is too large
        session_ids = (time_diffs > max_gap_minutes * 60).cumsum().fillna(0).astype(int)

        for _, session in group.groupby(session_ids):
            if len(session) < min_session_len:
                continue

            session = session.reset_index(drop=True)

            for i in range(len(session)):
                start_idx = max(0, i - window_size + 1)
                window = session.iloc[start_idx : i + 1]

                # Drop if label is missing
                label = session[LabeledDatasetFeatures.LABEL.feature_name].iloc[i]
                if pd.isna(label):
                    continue

                x_window = window[features].values
                x_padded = pad_sequences(
                    [x_window], maxlen=window_size, dtype="float32", padding="pre", value=-1
                )[0]

                X.append(x_padded)
                y.append(label)
                meta.append({
                    "cow_id": animal_id,
                    "timestamp": session[LabeledDatasetFeatures.DATETIME.feature_name].iloc[i]
                })

    logger.info(f"X length: {len(X)}")
    logger.info(f"y length: {len(y)}")

    return np.array(X), np.array(y), meta
