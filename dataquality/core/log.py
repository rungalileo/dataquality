import json
from typing import Dict

from dataquality import config
from dataquality.loggers import JsonlLogger


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger(config=config)


_logger = Logger()


def log(data: Dict) -> None:
    # TODO: logger_mode validation
    _logger.jsonl_logger.writer.write(data)
