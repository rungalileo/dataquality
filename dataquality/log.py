from typing import Dict

from dataquality.loggers import JsonlLogger


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


def log(data: Dict) -> None:
    logger = Logger()
