from typing import Dict

from galileo_python.loggers import JsonlLogger


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


def log(data: Dict) -> None:
    logger = Logger()
