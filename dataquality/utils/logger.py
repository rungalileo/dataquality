import logging
import os
import time

from dataquality.core._config import ConfigData, config

tz = time.strftime("%z")
logging.basicConfig(
    format=(f"[%(asctime)s.%(msecs)03d {tz}] " "[%(levelname)s] %(message)s"),
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger() -> logging.Logger:
    """Returns a logger based on the current run_id

    logger.getLogger caches loggers by name, so it won't return a new logging
    handler each time. We will have only 1 handler per new run
    """
    logger = logging.Logger(config.current_run_id)
    logger.setLevel(os.environ.get("GALILEO_LOG_LEVEL", logging.INFO).upper())
    galileo_home = ConfigData.DEFAULT_GALILEO_CONFIG_DIR
    handler = logging.FileHandler(
        os.path.join(f"{galileo_home}/stdout/{config.current_run_id}/stdout.log"), "a"
    )
    logger.addHandler(handler)
    return logger
