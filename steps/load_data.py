import pandas as pd
from zenml import step

from core.features import LabeledDatasetFeatures
from core.log import get_logger

logger = get_logger(name="data_preparation")


# @step
def load_data(
    data_path: str, sheet_name: str, delimiter: str, decimal: str
) -> pd.DataFrame:
    """
    Loads data from a CSV or Excel file into a Pandas DataFrame.

    Args:
        data_path (str): Path to the data file. Must be a .csv or .xlsx file.
        sheet_name (str): Name of the sheet to read from if the file is an Excel spreadsheet.
        delimiter (str): Delimiter used in the CSV file (e.g., ',' or ';').
        decimal (str): Character used for decimal separation in numeric values.

    Returns:
        pd.DataFrame: The loaded data.

    Raises:
        ValueError: If the file format is not supported.
        ValueError: If the file is not found.
        ValueError: If the loaded DataFrame is empty.
    """

    logger.info(f"Loading file {data_path}")

    try:
        if data_path.endswith("xlsx"):
            df = pd.read_excel(data_path, sheet_name=sheet_name or 0)
        elif data_path.endswith("csv"):
            df = pd.read_csv(data_path, delimiter=delimiter, decimal=decimal)
        else:
            raise ValueError(
                f"Unsupported file format: {data_path}. Expected .xlsx or .csv."
            )
    except FileNotFoundError:
        raise ValueError(f"File not found: {data_path}")

    if df.empty:
        logger.info("Dataframe is empty")
        raise ValueError("Dataframe is empty.")

    return df
