import pandas as pd
from zenml import step

from core.features import LabeledDatasetFeatures
from core.log import get_logger

logger = get_logger(__name__)


# @step
def engineer_features(df: pd.DataFrame, drop_original_features: bool):
    features_to_drop = []

    diff_features = {
        LabeledDatasetFeatures.ODOMETER_DIFF.feature_name: LabeledDatasetFeatures.ODOMETER_KM.feature_name,
        LabeledDatasetFeatures.IMU_40_DIFF.feature_name: LabeledDatasetFeatures.IMU_40.feature_name,
        LabeledDatasetFeatures.IMU_80_DIFF.feature_name: LabeledDatasetFeatures.IMU_80.feature_name,
        LabeledDatasetFeatures.IMU_120_DIFF.feature_name: LabeledDatasetFeatures.IMU_120.feature_name,
        LabeledDatasetFeatures.IMU_160_DIFF.feature_name: LabeledDatasetFeatures.IMU_160.feature_name,
        LabeledDatasetFeatures.IMU_200_DIFF.feature_name: LabeledDatasetFeatures.IMU_200.feature_name,
        LabeledDatasetFeatures.IMU_240_DIFF.feature_name: LabeledDatasetFeatures.IMU_240.feature_name,
    }

    # Observations of the same animal are may be interrupted. I.e., the same animal might be observed
    # for a few minutes today and a few minutes on another day. Calculating the difference between the
    # first observation after the "jump" and the last observaion before the "jump" is misleading.
    df["time_diff"] = df.groupby(LabeledDatasetFeatures.ANIMAL_ID.feature_name)[
        LabeledDatasetFeatures.DATETIME.feature_name
    ].diff()

    # If time_diff is larger than expected (e.g. >1 minute), start new session
    df["new_session"] = (df["time_diff"] > pd.Timedelta(minutes=1)).astype(int)

    df["session_id"] = df.groupby(LabeledDatasetFeatures.ANIMAL_ID.feature_name)[
        "new_session"
    ].cumsum()

    session_group_cols = [LabeledDatasetFeatures.ANIMAL_ID.feature_name, "session_id"]

    # Step 2: Compute deltas
    for diff_feature, original_feature in diff_features.items():
        if original_feature in df.columns:
            df[diff_feature] = df.groupby(session_group_cols)[original_feature].diff()
            # Only drop the original feature if it's the odometer
            
            if original_feature == LabeledDatasetFeatures.ODOMETER_KM.feature_name:
                features_to_drop.append(original_feature)
        else:
            logger.warning(
                f"Feature '{original_feature}' not found in dataframe. Skipping diff for '{diff_feature}'."
            )

    # Step 3: Compute rolling stats (e.g., sum and mean over 5-minute window)
    rolling_window = 20  # Assuming 1-minute intervals
    for diff_feature in diff_features.keys():
        if diff_feature in df.columns:
            df[f"{diff_feature}_rolling_sum"] = df.groupby(session_group_cols)[
                diff_feature
            ].transform(lambda x: x.rolling(rolling_window, min_periods=1).sum())
            df[f"{diff_feature}_rolling_mean"] = df.groupby(session_group_cols)[
                diff_feature
            ].transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())

    # Optional: drop originals
    df.drop(
        columns=[
            "time_diff",
            "new_session",
            "session_id",
            "Odometer_km"
        ],
        inplace=True,
    )

    if drop_original_features:
        df.drop(columns=features_to_drop, inplace=True)

    return df
