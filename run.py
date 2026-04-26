import os
import warnings

import click
from dotenv import load_dotenv

from core.log import get_logger
from pipelines.config import (
    ClassifierPipelineConfig,
    LogoPipelineConfig,
    LSTMPipelineConfig,
    TrainOnAllDataConfig,
    load_config,
)
from pipelines.train_classifier import train_classifier
from pipelines.train_logo import train_logo
from pipelines.train_lstm import train_lstm
from pipelines.train_on_all_data import train_on_all_data

load_dotenv()

logger = get_logger(name="runner")

warnings.filterwarnings("ignore")


@click.command()
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(
        ["train_classifier", "train_lstm", "train_logo", "train_on_all_data"],
        case_sensitive=False,
    ),
    default="train_classifier",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional config path override for the selected pipeline.",
)
def main(
    pipeline: str = "train_classifier",
    config: str | None = None,
):
    """
    Execute one configured training pipeline.

    Args:
        pipeline: Pipeline name.
        config: Optional path to YAML config. If omitted, pipeline defaults are used.
    """

    if pipeline == "train_classifier":
        config_path = config or "./configs/pipelines/train_classifier.yaml"
        train_classifier(load_config(ClassifierPipelineConfig, config_path))

    elif pipeline == "train_lstm":
        config_path = config or "./configs/pipelines/train_lstm.yaml"
        train_lstm(load_config(LSTMPipelineConfig, config_path))
    elif pipeline == "train_logo":
        config_path = config or "./configs/pipelines/train_logo.yaml"
        train_logo(load_config(LogoPipelineConfig, config_path))
    elif pipeline == "train_on_all_data":
        config_path = config or "./configs/pipelines/train_on_all_data.yaml"
        train_on_all_data(load_config(TrainOnAllDataConfig, config_path))
    else:
        raise ValueError(f"Pipeline `{pipeline}` does not exist.")

    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
