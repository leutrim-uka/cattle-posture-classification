from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated
from zenml import step

from core.experiments import get_experiment_features
from core.features import FeatureDataTypes, LabeledDatasetFeatures
from core.log import get_logger
from core.utils import convert_to_float

logger = get_logger(name="data_preparation")


# @step
def prepare_eShepherd_data(
    df: pd.DataFrame,
    experiment_number: int,
    missing_strategy: str,
    remove_outliers: bool,
    feature_engineering: bool = False,
    is_lstm: bool = False,
) -> pd.DataFrame:
    temporary_shape = df.shape
    logger.debug(f"Shape before cleaning: {temporary_shape}")

    # Select features based on experiment number
    df = feature_selection(df, experiment_number, is_lstm)

    if missing_strategy == "drop":
        df = df.dropna()
        logger.debug(f"Shape after dropping nulls: {df.shape}")
        logger.debug(f"{temporary_shape[0] - df.shape[0]} rows dropped.")
        temporary_shape = df.shape


    df = ensure_correct_datatype(df)

    df = remove_zero_imu_ticks(df)

    """When we used IMU Tick 240 to calculate outliers based on the label (separately for lying and standing),
    we found that some readings fall outside the 1.5*IQR range. We experimented with dropping these rows, and
    it resulted in a huge model performance increase. However, this is a questionable approach, as it 
    artificially makes the task easier by removing hard-to-classify samples.
    """
    if remove_outliers:
        df = remove_outlier_rows(df)
        logger.debug(f"Shape after removing outliers: {df.shape}")
        logger.debug(f"{temporary_shape[0] - df.shape[0]} rows dropped.")
        temporary_shape = df.shape

    # Create label feature
    df.loc[:, LabeledDatasetFeatures.LABEL.feature_name] = generate_label(
        df, LabeledDatasetFeatures.LABEL.feature_name
    )

    # Calculate the age of the animals if the birthday feature is included in the processing
    if LabeledDatasetFeatures.BIRTHDAY.feature_name in df.columns:
        df = calculate_age(df)

    # Comment this line out if the feature engineering is performed
    if not feature_engineering:
        df = compute_diff_features(df)

    logger.debug(f"Shape after cleaning: {df.shape}")

    return df


# @step
def grouped_train_test_split(
    df: pd.DataFrame,
    random_state: int,
    test_size: float,
) -> Tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
]:

    label_col = LabeledDatasetFeatures.LABEL.feature_name
    id_col = LabeledDatasetFeatures.ANIMAL_ID.feature_name

    # Identify cows with only 'Stehen'
    # no_liegen_ids = df.groupby(id_col)[label_col].apply(lambda x: (x == 0).sum() == 0)
    # no_liegen_ids = no_liegen_ids[no_liegen_ids].index.tolist()

    # Candidate cows excluding standing-only
    # candidate_ids = [uid for uid in df[id_col].unique() if uid not in no_liegen_ids]

    # Per-cow proportion of Liegen for stratification (rounded for stability)
    # Compute raw proportions
    liegen_props = (
        df
        .groupby(id_col)[label_col]
        .apply(lambda x: (x == 0).mean())
    )

    liegen_props = liegen_props.fillna(0)

    # Use qcut for balanced bins
    cow_bins = pd.qcut(liegen_props, q=6, labels=False, duplicates='drop')

    bin_counts = cow_bins.value_counts()

    for bin_idx, count in bin_counts.items():
        logger.info(f"Bin {bin_idx}: {count} cows")

    if (len(liegen_props) < 4) or (bin_counts < 2).any():
        logger.warning("Not enough cows per bin — falling back to non-stratified split.")
        train_ids, test_ids = train_test_split(
            liegen_props.index,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )
    else:
        train_ids, test_ids = train_test_split(
            liegen_props.index,
            test_size=test_size,
            random_state=random_state,
            stratify=cow_bins,
        )

    # Masks for train/test
    train_mask = df[id_col].isin(train_ids)
    test_mask = df[id_col].isin(test_ids)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Test df shape: {test_df.shape}")
    logger.info(f"Train cows: {len(train_ids)} | Test cows: {len(test_ids)}")
    logger.info(f"Liegen proportion train: {(train_df[label_col] == 0).mean():.3f}")
    logger.info(f"Liegen proportion test: {(test_df[label_col] == 0).mean():.3f}")
    logger.info(f"IDs in test set: {test_df.animal_id.unique().tolist()}")

    return train_df, test_df


def random_train_test_split(
    df: pd.DataFrame,
    random_state: int,
    test_size: float,
) -> Tuple[
    Annotated[pd.DataFrame, "train_df"],
    Annotated[pd.DataFrame, "test_df"],
]:
    """Random row-level split with label stratification.

    This intentionally allows the same animal_id to appear in both train and test.
    """
    label_col = LabeledDatasetFeatures.LABEL.feature_name

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    id_col = LabeledDatasetFeatures.ANIMAL_ID.feature_name
    train_ids = set(train_df[id_col].astype(str).unique()) if id_col in train_df.columns else set()
    test_ids = set(test_df[id_col].astype(str).unique()) if id_col in test_df.columns else set()
    overlap_count = len(train_ids.intersection(test_ids)) if train_ids and test_ids else 0

    logger.info(f"Train df shape: {train_df.shape}")
    logger.info(f"Test df shape: {test_df.shape}")
    logger.info(f"Liegen proportion train: {(train_df[label_col] == 0).mean():.3f}")
    logger.info(f"Liegen proportion test: {(test_df[label_col] == 0).mean():.3f}")
    logger.info(f"Train/Test animal overlap count (expected >0 for random split): {overlap_count}")

    return train_df, test_df


def feature_selection(df: pd.DataFrame, experiment_number: list, is_lstm: bool):

    # We predefine features to keep based on experiment number (core/experiments.py)
    features_to_keep = get_experiment_features(experiment_number=experiment_number)

    if is_lstm:
        # features_to_keep.extend([LabeledDatasetFeatures.DATETIME.feature_name])
        pass

    df = df[features_to_keep]
    logger.info(f"Kept features: {df.columns}")
    return df


def calculate_age(df: pd.DataFrame):
    df[LabeledDatasetFeatures.AGE.feature_name] = (
        df[LabeledDatasetFeatures.DATETIME.feature_name]
        - pd.to_datetime(df[LabeledDatasetFeatures.BIRTHDAY.feature_name])
    ).dt.days // 365

    df.drop(columns=[LabeledDatasetFeatures.BIRTHDAY.feature_name], inplace=True)
    return df


def generate_label(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Encodes posture labels as binary values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column_name (str): Column containing posture labels.

    Returns:
        pd.Series: Column with binary-encoded labels.
    """
    return df[column_name].map(lambda x: 0 if x == "Liegen" else 1)


def ensure_correct_datatype(df: pd.DataFrame) -> pd.DataFrame:
    # Numerical features -> float
    for feature in df.columns:
        logger.info(f"Ensuring correct datatype for feature: {feature}")
        logger.info(f"Feature dtype before: {df[feature].dtype}")
        if feature in LabeledDatasetFeatures.get_feature_by_type(
            FeatureDataTypes.NUMERICAL_STD_SCALER, FeatureDataTypes.NUMERICAL_MIN_MAX
        ):
            df[feature] = df[feature].apply(convert_to_float)
        elif feature in LabeledDatasetFeatures.get_feature_by_type(
            FeatureDataTypes.DATETIME_META, FeatureDataTypes.DATETIME_FEAT
        ):
            df[feature] = pd.to_datetime(df[feature], errors="coerce").dt.tz_localize(
                None
            )
        elif feature in LabeledDatasetFeatures.get_feature_by_type(
            FeatureDataTypes.ONE_HOT, FeatureDataTypes.STRING_ID
        ):
            df[feature] = df[feature].astype("string")
        else:
            logger.warning(f"Feature {feature} is not supposed to be in the dataset!")

    return df


def remove_outlier_rows(df: pd.DataFrame) -> pd.DataFrame:
    imu_col = "IMU_Tick_Count_240mG"

    for label in ["Liegen", "Stehen"]:
        subset = df[df[LabeledDatasetFeatures.LABEL.feature_name] == label]
        Q1 = subset[imu_col].quantile(0.25)
        Q3 = subset[imu_col].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"{label}: removing outliers outside [{lower_bound:.2f}, {upper_bound:.2f}]")

        df = df[~(
            (df[LabeledDatasetFeatures.LABEL.feature_name] == label) &
            ((df[imu_col] < lower_bound) | (df[imu_col] > upper_bound))
        )]

    return df



def remove_zero_imu_ticks(df: pd.DataFrame) -> pd.DataFrame:
    """In some rows, all IMU Tick features contain zeros. Even if the label is "standing" or "walking".
    As such, we assume all-zero readings to be faulty and drop them.
    """
    imu = [
        LabeledDatasetFeatures.IMU_40.feature_name, LabeledDatasetFeatures.IMU_80.feature_name,LabeledDatasetFeatures.IMU_120.feature_name,LabeledDatasetFeatures.IMU_160.feature_name,LabeledDatasetFeatures.IMU_200.feature_name,LabeledDatasetFeatures.IMU_240.feature_name
        ]
    logger.debug(f"Shape before removing all zeros: {df.shape}")
    temporary_shape = df.shape
    df = df[df[imu].sum(axis=1) != 0]
    logger.debug(f"{temporary_shape[0] - df.shape[0]} rows dropped.")
    return df


def compute_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    diff_features = {
        LabeledDatasetFeatures.ODOMETER_DIFF.feature_name: LabeledDatasetFeatures.ODOMETER_KM.feature_name,
        # add others if needed later
    }

    # Prepare session grouping
    df["time_diff"] = df.groupby(LabeledDatasetFeatures.ANIMAL_ID.feature_name)[
        LabeledDatasetFeatures.DATETIME.feature_name
    ].diff()
    # We consider a new session if time difference > 4 minutes
    df["new_session"] = (df["time_diff"] > pd.Timedelta(minutes=4)).astype(int)
    df["session_id"] = df.groupby(LabeledDatasetFeatures.ANIMAL_ID.feature_name)["new_session"].cumsum()
    session_group_cols = [LabeledDatasetFeatures.ANIMAL_ID.feature_name, "session_id"]

    # Compute differences per session
    for diff_feature, original_feature in diff_features.items():
        if original_feature in df.columns:
            df[diff_feature] = df.groupby(session_group_cols)[original_feature].diff()
        else:
            logger.warning(
                f"Feature '{original_feature}' not found in dataframe. Skipping diff for '{diff_feature}'."
            )

    # Cleanup helper columns
    df.drop(columns=["time_diff", "new_session", "session_id", LabeledDatasetFeatures.ODOMETER_KM.feature_name], inplace=True)
    logger.info(df.columns)
    return df