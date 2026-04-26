import warnings
from typing import Optional, Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import ArtifactConfig, step

from core.encoding import encode_features
from core.features import LabeledDatasetFeatures
from core.log import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")


def _detect_imu_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect IMU columns including rolling stats (as long as names contain 'IMU').
    Adjust this if your rolling feature names don't contain 'IMU'.
    """
    imu_cols = [c for c in df.columns if "IMU" in c]
    logger.info(f"Detected {len(imu_cols)} IMU-related columns for normalization.")
    logger.info(f"IMU columns: {imu_cols}")
    return imu_cols


def cow_robust_normalize_imu(
    df: pd.DataFrame,
    cow_col: str,
    imu_cols: list[str],
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Per-cow robust normalization:
        x' = (x - median_cow) / IQR_cow

    - Uses only X (no labels)
    - Safe for LOGO if applied separately to train_df and test_df
    """
    if not imu_cols:
        return df

    out = df.copy()
    g = out.groupby(cow_col, sort=False)

    med = g[imu_cols].transform("median")
    q75 = g[imu_cols].transform(lambda s: s.quantile(0.75))
    q25 = g[imu_cols].transform(lambda s: s.quantile(0.25))
    iqr = (q75 - q25).astype(float)

    # avoid division by zero / constant features within a cow
    iqr = iqr.mask(iqr.abs() < eps, 1.0)

    # impute NaNs with cow median then normalize
    out[imu_cols] = out[imu_cols].fillna(med)
    out[imu_cols] = (out[imu_cols] - med) / iqr

    return out


def feature_transformation(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
    feature_engineering: bool = False,
    cow_normalize_imu: bool = False,
) -> Tuple[pd.DataFrame, dict, dict]:
    """Transform and encode features of the dataset.

    Args:
        df (pd.DataFrame): Dataset.
        merge_textual_features (bool): Indicator
        encode_labels (bool): Indicator
        encoders (dict): Encoders

    Returns:
        np.ndarray: Encoded dataset.
        dict: Encoders used for transformation.
        dict: Vector size of each encoded feature.
    """

    df_in = df.copy()

    if cow_normalize_imu:
        cow_col = LabeledDatasetFeatures.ANIMAL_ID.feature_name
        imu_cols = _detect_imu_columns(df_in)
        df_in = cow_robust_normalize_imu(df_in, cow_col=cow_col, imu_cols=imu_cols)

    encoded_df, encoders, vector_sizes = encode_features(
        df_in,
        encoders=encoders,
        feature_engineering=feature_engineering
    )
    return encoded_df, encoders, vector_sizes
