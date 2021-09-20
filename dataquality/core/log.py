from typing import Dict, List

from pydantic.error_wrappers import ValidationError

from dataquality import config
from dataquality.loggers import JsonlLogger
from dataquality.schemas.jsonl_logger import JsonlInputLogItem, JsonlOutputLogItem


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


logger = Logger()


def log_input_data(data: Dict) -> None:
    try:
        input_data = JsonlInputLogItem(**data)
    except ValidationError as e:
        raise e
    assert config.current_project_id is not None
    assert config.current_run_id is not None
    logger.jsonl_logger.write_input(
        config.current_project_id, config.current_run_id, input_data.dict()
    )


def log_model_output(data: Dict) -> None:
    try:
        output_data = JsonlOutputLogItem(**data)
    except ValidationError as e:
        raise e
    assert config.current_project_id is not None
    assert config.current_run_id is not None
    logger.jsonl_logger.write_output(
        config.current_project_id, config.current_run_id, output_data.dict()
    )


def set_labels(labels: List[str]) -> None:
    """
    Creates the mapping of the labels for the model to their respective indexes.

    :param labels: An ordered list of labels (ie ['dog','cat','fish']
    :return: None
    """
    config.labels = labels
