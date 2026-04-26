import warnings
from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from core.features import FeatureDataTypes, LabeledDatasetFeatures, SmaxtecFeatures
from core.log import get_logger
from core.utils import drop_unnecessary_features

logger = get_logger(name="ai4mplus.features")
warnings.filterwarnings("ignore")


def _encode_one_hot_categorical(
    series: pd.Series,
    encoder: Optional[OneHotEncoder] = None,
    nan_strategy: Literal["missing"] = "missing",
    impute_value=None
) -> Tuple[pd.Series, OneHotEncoder]:
    """Encode categorical features.

    Args:
        series (pd.Series): Series to encode.
        nan_strategy (str): Strategy to handle missing values. Currently only "missing" is supported.
            - "missing": Fill missing values with "missing"
        encoder (OneHotEncoder): Encoder to use for encoding (optional). If not provided, a new encoder will be created.

    Returns:
        pd.Series: Encoded series.
        OneHotEncoder: Encoder used for encoding.
    """
    if nan_strategy == "missing":
        impute_value = "missing"
    elif nan_strategy == "mode":
        if impute_value is None:  # training phase
            impute_value = series.mode(dropna=True).iloc[0] if not series.mode(dropna=True).empty else "missing"

    # Apply imputation
    series = series.fillna(impute_value)

    if encoder:
        logger.info("Using preset scaler!")
        transformed_series = encoder.transform(series.to_frame())
    else:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transformed_series = encoder.fit_transform(series.to_frame())

    encoded_df = pd.DataFrame(
        transformed_series,
        columns=encoder.get_feature_names_out([series.name]),
        index=series.index,
    )

    return encoded_df, encoder, impute_value


def _encode_numerical(
    series: pd.Series,
    scaler_type: str,
    scaler: Optional[Union[StandardScaler,]] = None,
    apply_log_transformer: bool = False,
    nan_strategy: Literal["mean", "zero"] = "mean",
    impute_value=None,
) -> Tuple[pd.Series, StandardScaler]:
    """Encode numerical features with a StandartScale.

    Args:
        series (pd.Series): Series to encode.
        nan_strategy (str): Strategy to handle missing values.
        scaler (StandardScaler): Scaler to use for encoding (optional). If not provided, a new scaler will be created.

    Returns:
        pd.Series: Encoded series.
        StandardScaler: Scaler used for encoding.
    """

    # assert series.dtype in [int, float], "Series must be numerical"
    def _log_transform(x):
        """Apply log1p transformation to the series."""
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log1p(x)

        x = np.where(np.isneginf(x) | np.isnan(x), 0, x)
        return pd.Series(x)

    if impute_value is None:
        if nan_strategy == "mean":
            impute_value = series.mean()
        else:
            impute_value = 0

    series = series.fillna(impute_value)

    if apply_log_transformer:
        series = _log_transform(series)

    if scaler:
        logger.info("Using predefined scaler")
        series = pd.Series(
            scaler.transform(series.values.reshape(-1, 1)).squeeze(), index=series.index
        )
    else:
        scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()
        series = pd.Series(
            scaler.fit_transform(series.values.reshape(-1, 1)).squeeze(),
            index=series.index,
        )

    assert len(series[series.isna()]) == 0, (
        "Numerical features cannot have missing values"
    )

    return series, scaler, impute_value


def encode_features(
    df: pd.DataFrame,
    encoders: Optional[dict] = None,
    feature_engineering: bool = False,
) -> Tuple[pd.DataFrame, dict, dict]:
    encoders = {} if encoders is None else encoders
    vector_sizes = {}
    impute_values = {}

    new_df = df.copy()

    all_features = list(LabeledDatasetFeatures.get_non_optional_features(return_names=False)) \
                   + list(SmaxtecFeatures)

    for feature in all_features:
        if feature.feature_name not in df.columns:
            logger.warning(
                f"Skipping feature '{feature.feature_name}': Not in dataframe!"
            )
            continue

        if feature.feature_name == LabeledDatasetFeatures.LABEL.feature_name:
            continue

        encoder = None
        vector_size = None
        impute_value = None

        series = new_df[feature.feature_name]

        if feature.data_type == FeatureDataTypes.NUMERICAL_MIN_MAX:
            logger.info(f"Encoding numerical: {feature.feature_name}")
            series, encoder, impute_value = _encode_numerical(
                series,
                scaler_type="minmax",
                scaler=encoders.get(feature.feature_name),
                apply_log_transformer=False,
                nan_strategy="else",
                impute_value=impute_values.get(feature.feature_name),
            )
            vector_size = 1
            series.index = new_df.index
            new_df[feature.feature_name] = series
        if feature.data_type == FeatureDataTypes.NUMERICAL_STD_SCALER:
            logger.info(f"Encoding numerical: {feature.feature_name}")
            series, encoder, impute_value = _encode_numerical(
                series,
                scaler_type="std",
                scaler=encoders.get(feature.feature_name),
                apply_log_transformer=False,
                nan_strategy="else",
                impute_value=impute_values.get(feature.feature_name),
            )
            vector_size = 1
            series.index = new_df.index
            new_df[feature.feature_name] = series
        elif feature.data_type == FeatureDataTypes.ONE_HOT:
            logger.info(f"Encoding one hot: {feature.feature_name}")
            encoded_df, encoder, impute_value = _encode_one_hot_categorical(
                series,
                encoder=encoders.get(feature.feature_name),
                nan_strategy="missing",
                impute_value=impute_values.get(feature.feature_name)
            )
            new_df = pd.concat([new_df, encoded_df], axis=1)
            new_df.drop(columns=[feature.feature_name], inplace=True)
            vector_size = encoded_df.shape[1]
        elif feature.data_type in [
            # FeatureDataTypes.DATETIME_META,
            FeatureDataTypes.DATETIME_FEAT,
        ]:
            if feature_engineering:
                pass
            else:
                new_df = drop_unnecessary_features(
                    new_df, "encoding", [feature.feature_name]
                )
                continue

        # Update the encoders and vector sizes
        if encoder:
            encoders[feature.feature_name] = encoder
        if vector_size:
            vector_sizes[feature.feature_name] = vector_size
        if impute_value:
            impute_values[feature.feature_name] = impute_value


    return new_df, encoders, vector_sizes
