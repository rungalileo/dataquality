import logging
import os
from typing import Any, Tuple

from dataquality.core._config import ConfigData, config

STDOUT_HOME = f"{ConfigData.DEFAULT_GALILEO_CONFIG_DIR}/stdout"


class CustomSplitAdapter(logging.LoggerAdapter):
    """
    This adapter appends the split to the message, if found. Otherwise, "None"

    Adapted from https://docs.python.org/3/howto/logging-cookbook.html (CustomAdapter)
    """

    def process(self, msg: str, kwargs: Any) -> Tuple[str, Any]:
        split = kwargs.pop("split", self.extra["split"])
        epoch = kwargs.pop("epoch", self.extra["epoch"])
        if epoch is not None:
            return "[%s] [epoch:%s]: %s" % (split, str(epoch), msg), kwargs
        return "[%s]: %s" % (split, msg), kwargs


log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
)


def get_stdout_logger() -> CustomSplitAdapter:
    """Returns a logger based on the current run_id"""
    logger = logging.getLogger(str(config.current_run_id))
    logger.setLevel(os.environ.get("GALILEO_LOG_LEVEL", "INFO").upper())
    # Avoid adding multiple handlers if one already exists
    if not logger.handlers:
        handler = logging.FileHandler(
            f"{STDOUT_HOME}/{config.current_run_id}/stdout.log"
        )
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    adapter = CustomSplitAdapter(logger, {"split": None, "epoch": None})
    return adapter
