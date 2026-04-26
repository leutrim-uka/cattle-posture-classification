import logging


def get_logger(name: str = "moomotion", level: int = logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger
