from enum import Enum


class FeatureDataTypes(Enum):
    DATETIME_META = "datetime_metadata"
    DATETIME_FEAT = "datetime_feature"
    NUMERICAL_MIN_MAX = "numerical_min_max"
    NUMERICAL_STD_SCALER = "numerical_std"
    COORDINATES = "coordinates"
    ONE_HOT = "one_hot"
    BOOLEAN = "boolean"
    STRING_ID = "string_id"
    LIST = "list"
    OPTIONAL = "optional"


class LabeledDatasetFeatures(Enum):
    ANIMAL_ID = ("animal_id", FeatureDataTypes.STRING_ID)
    DATETIME = ("datetime", FeatureDataTypes.DATETIME_META)
    LABEL = ("Stehen/Liegen", FeatureDataTypes.ONE_HOT)
    ODOMETER_KM = ("Odometer_km", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_40 = ("IMU_Tick_Count_40mG", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_80 = ("IMU_Tick_Count_80mG", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_120 = ("IMU_Tick_Count_120mG", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_160 = ("IMU_Tick_Count_160mG", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_200 = ("IMU_Tick_Count_200mG", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_240 = ("IMU_Tick_Count_240mG", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # Herde Liste Features
    BIRTHDAY = ("GEB_DATR", FeatureDataTypes.DATETIME_FEAT)
    GENDER = ("GESCHL_R", FeatureDataTypes.ONE_HOT)
    HERDE = ("Herde", FeatureDataTypes.ONE_HOT)
    BREED = ("Breed", FeatureDataTypes.ONE_HOT)

    # Engineered features
    AGE = ("Age", FeatureDataTypes.NUMERICAL_MIN_MAX)
    IMU_TOTAL_WEIGHTED = ("IMU_total", FeatureDataTypes.NUMERICAL_MIN_MAX)

    IMU_40_DIFF = ("IMU_Tick_Count_40mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_80_DIFF = ("IMU_Tick_Count_80mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_120_DIFF = ("IMU_Tick_Count_120mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_160_DIFF = ("IMU_Tick_Count_160mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_200_DIFF = ("IMU_Tick_Count_200mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_240_DIFF = ("IMU_Tick_Count_240mG_diff", FeatureDataTypes.NUMERICAL_STD_SCALER)

    IMU_40_DIFF_ROLLING_SUM = ("IMU_Tick_Count_40mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_80_DIFF_ROLLING_SUM = ("IMU_Tick_Count_80mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_120_DIFF_ROLLING_SUM = ("IMU_Tick_Count_120mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_160_DIFF_ROLLING_SUM = ("IMU_Tick_Count_160mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_200_DIFF_ROLLING_SUM = ("IMU_Tick_Count_200mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_240_DIFF_ROLLING_SUM = ("IMU_Tick_Count_240mG_diff_rolling_sum", FeatureDataTypes.NUMERICAL_STD_SCALER)

    IMU_40_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_40mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_80_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_80mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_120_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_120mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_160_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_160mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_200_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_200mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)
    IMU_240_DIFF_ROLLING_MEAN = ("IMU_Tick_Count_240mG_diff_rolling_mean", FeatureDataTypes.NUMERICAL_STD_SCALER)

    ACTIVITY = ("act", FeatureDataTypes.NUMERICAL_MIN_MAX)
    TEMPERATURE = ("temp", FeatureDataTypes.NUMERICAL_STD_SCALER)
    TEMPERATURE_NORMAL_INDEX = ("temp_normal_index", FeatureDataTypes.NUMERICAL_STD_SCALER)
    HEAT_INDEX = ("heat_index", FeatureDataTypes.NUMERICAL_STD_SCALER)
    CALVING_INDEX = ("calving_index", FeatureDataTypes.NUMERICAL_MIN_MAX)
    RUMINATING_INDEX = ("rum_index", FeatureDataTypes.NUMERICAL_MIN_MAX)
    WATER_INTAKE = ("water_intake", FeatureDataTypes.NUMERICAL_MIN_MAX)
    CLIMATE_TEMPERATURE = ("climate_temp", FeatureDataTypes.NUMERICAL_STD_SCALER)
    CLIMATE_HUMIDITY = ("climate_hum", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # THERMAL COMFORT (highest value)
    AVG_AIR_TEMP = ("AirT_C_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)
    AVG_WET_BULB_TEMP = ("Twetbulb_C_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)
    DEW_POINT_TEMP = ("Tdewpt_C_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)
    REL_HUMIDITY = ("RelHumid", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # GROUND COMFORT
    AVG_GROUND_TEMP = ("GroundT_C_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)
    AVG_SOIL_WATER_CONTENT = ("VWC_C_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # SOLAR LOAD
    AVG_SHORTWAVE_INCOMING = ("Rad_SWin_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)
    POTENTIAL_SOLAR_RADIATION = ("PotSlrRad_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # WIND (secondary modifier)
    WIND_SPEED = ("WindSpd_m_s_Avg", FeatureDataTypes.NUMERICAL_MIN_MAX)

    # PRECIPITATION (contextual)
    RAINFALL = ("Rain_mm_Tot", FeatureDataTypes.NUMERICAL_MIN_MAX)


    # Engineered cumulative features
    ODOMETER_DIFF = ("Odometer_Diff", FeatureDataTypes.NUMERICAL_MIN_MAX)

    ODOMETER_DIFF_ROLLING_SUM = (
        "Odometer_Diff_rolling_sum",
        FeatureDataTypes.NUMERICAL_MIN_MAX,
    )

    ODOMETER_DIFF_ROLLING_MEAN = (
        "Odometer_Diff_rolling_mean",
        FeatureDataTypes.NUMERICAL_MIN_MAX,
    )



    def __new__(self, column_name, data_type):
        obj = object.__new__(self)
        obj._value_ = column_name
        obj.feature_name = column_name
        obj.data_type = data_type
        return obj

    @staticmethod
    def get_all_features(return_names: bool = True) -> list:
        features = [feature for feature in LabeledDatasetFeatures]

        if return_names:
            features = [feature.feature_name for feature in features]

        return features

    @staticmethod
    def get_imu_features(return_names: bool = True) -> list:
        features = [
            feature
            for feature in LabeledDatasetFeatures
            if "IMU" in feature.feature_name
        ]

        if return_names:
            features = [feature.feature_name for feature in features]

        return features

    @staticmethod
    def get_optional_features(return_names: bool = True) -> list:
        features = [
            feature
            for feature in LabeledDatasetFeatures
            if feature.data_type == FeatureDataTypes.OPTIONAL
        ]
        if return_names:
            features = [feature.feature_name for feature in features]

        return features

    @staticmethod
    def get_non_optional_features(return_names: bool = True) -> list:
        features = [
            feature
            for feature in LabeledDatasetFeatures
            if feature.data_type != FeatureDataTypes.OPTIONAL
        ]
        if return_names:
            features = [feature.feature_name for feature in features]

        return features

    @staticmethod
    def get_feature_by_type(
        *data_type: FeatureDataTypes,
        return_names: bool = True,
    ) -> list:
        features = [
            feature
            for feature in LabeledDatasetFeatures
            if feature.data_type in data_type
        ]

        if return_names:
            features = [feature.feature_name for feature in features]

        return features


class SmaxtecFeatures(Enum):
    ANIMAL_ID = ("animal_id", FeatureDataTypes.STRING_ID)
    TIMESTAMP = ("timestamp", FeatureDataTypes.DATETIME_META)
    ACTIVITY = ("act", FeatureDataTypes.NUMERICAL_MIN_MAX)
    TEMPERATURE = ("temp", FeatureDataTypes.NUMERICAL_STD_SCALER)
    TEMPERATURE_NORMAL_INDEX = ("temp_normal_index", FeatureDataTypes.NUMERICAL_STD_SCALER)
    HEAT_INDEX = ("heat_index", FeatureDataTypes.NUMERICAL_STD_SCALER)
    CALVING_INDEX = ("calving_index", FeatureDataTypes.NUMERICAL_MIN_MAX)
    RUMINATING_INDEX = ("rum_index", FeatureDataTypes.NUMERICAL_MIN_MAX)
    WATER_INTAKE = ("water_intake", FeatureDataTypes.NUMERICAL_MIN_MAX)
    CLIMATE_TEMPERATURE = ("climate_temp", FeatureDataTypes.NUMERICAL_STD_SCALER)
    CLIMATE_HUMIDITY = ("climate_hum", FeatureDataTypes.NUMERICAL_MIN_MAX)

    def __new__(self, column_name, data_type):
        obj = object.__new__(self)
        obj._value_ = column_name
        obj.feature_name = column_name
        obj.data_type = data_type
        return obj
