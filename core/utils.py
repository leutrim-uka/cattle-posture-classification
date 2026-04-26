from typing import Optional

import pandas as pd

from core.features import LabeledDatasetFeatures


def convert_to_int(x: any) -> Optional[int]:
    """Convert a value to an integer.
    Args:
        x: Value to convert.
    Returns:
        int: Integer value or None if the value cannot be converted.
    """
    if x is None:
        return None

    try:
        return int(x)
    except ValueError:
        return None


def convert_to_float(x: any) -> Optional[int]:
    """Convert a value to an integer.
    Args:
        x: Value to convert.
    Returns:
        int: Integer value or None if the value cannot be converted.
    """
    if x is None:
        return None

    try:
        return float(x)
    except ValueError:
        return None


def drop_unnecessary_features(
    df: pd.DataFrame, stage: str, features: list = []
) -> pd.DataFrame:
    if stage == "encoding":
        df = df.drop(columns=features)
    elif stage == "training":
        df = df.drop()
    elif stage == "feature_engineering":
        pass
    elif stage == "splitting":
        # df = df.drop(columns=[LabeledDatasetFeatures.ANIMAL_ID.feature_name])
        pass

    return df
