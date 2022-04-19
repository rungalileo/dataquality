import logging
import os
import time

from dataquality.core._config import ConfigData, config

tz = time.strftime("%z")
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(threadName)s]: %(message)s"
)


def get_stdout_logger() -> logging.Logger:
    """Returns a logger based on the current run_id

    logger.getLogger caches loggers by name, so it won't return a new logging
    handler each time. We will have only 1 handler per new run
    """
    logger = logging.Logger(config.current_run_id)
    logger.setLevel(os.environ.get("GALILEO_LOG_LEVEL", "INFO").upper())
    galileo_home = ConfigData.DEFAULT_GALILEO_CONFIG_DIR
    handler = logging.FileHandler(
        f"{galileo_home}/stdout/{config.current_run_id}/stdout.log"
    )
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)
    return logger
