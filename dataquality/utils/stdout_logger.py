import logging
import os

from dataquality.core._config import ConfigData, config

STDOUT_HOME = f"{ConfigData.DEFAULT_GALILEO_CONFIG_DIR}/stdout"
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(threadName)s]: %(message)s"
)


def get_stdout_logger() -> logging.Logger:
    """Returns a logger based on the current run_id"""
    logger = logging.getLogger(str(config.current_run_id))
    logger.setLevel(os.environ.get("GALILEO_LOG_LEVEL", "INFO").upper())
    # Avoid adding multiple handlers if it already exists
    if not logger.hasHandlers():
        handler = logging.FileHandler(
            f"{STDOUT_HOME}/{config.current_run_id}/stdout.log"
        )
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    return logger
