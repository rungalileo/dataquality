from threading import Thread
from typing import Dict

import numpy as np
from pydantic.error_wrappers import ValidationError

from dataquality import config
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.exceptions import GalileoException
from dataquality.loggers import JsonlLogger
from dataquality.schemas.jsonl_logger import JsonlInputLogItem, JsonlOutputLogItem
from dataquality.schemas.split import Split


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


logger = Logger()


def log_input_data(data: Dict) -> None:
    """
    Function to log a single line of input data for a train/test/validation dataset.
    Use the log_batch_input_data instead to take advantage of threading.

    :param data: Dictionary of data attributes (input text, labels, and ids)
    :return: None
    """
    try:
        input_data = JsonlInputLogItem(**data)
    except ValidationError as e:
        raise e
    assert config.current_project_id is not None
    assert config.current_run_id is not None
    logger.jsonl_logger.write_input(
        config.current_project_id, config.current_run_id, input_data.dict()
    )


def _log_batch_input_data(data: GalileoDataConfig) -> None:
    """
    Threaded batch logger for Galileo input data. Used as target for
    log_batch_input_data

    :param data: GalileoDataConfig
    :return: None
    """
    try:
        data.validate()
    except AssertionError as e:
        raise GalileoException(e)
    ids = data.ids if data.ids else range(len(data.text))
    for idx, text, label in zip(ids, data.text, data.labels):
        log_input_data(
            {
                "id": idx,
                "text": text,
                "gold": str(label) if data.split != Split.inference else None,
                "split": data.split,
            }
        )


def log_batch_input_data(data: GalileoDataConfig) -> None:
    """
    First class function to log all input data in batch for a training/validation/test
    batch. Use log_batch_input_data instead of log_input_data to take advantage of
    multithreading and other validation support.

    :param data: GalileoDataConfig
    :return: None
    """
    try:
        in_thread = Thread(target=_log_batch_input_data, args=[data])
        in_thread.start()
    except Exception as e:
        raise GalileoException(e)


def log_model_output(data: Dict) -> None:
    """
    Function to log a single model output for a train/test/validation dataset.
    Use the log_model_outputs instead to take advantage of threading.

    :param data: Dictionary of model output (id, split, epoch, embeddings,
    probabilities and prediction)
    :return: None
    """
    try:
        output_data = JsonlOutputLogItem(**data)
    except ValidationError as e:
        raise e
    assert config.current_project_id is not None
    assert config.current_run_id is not None

    logger.jsonl_logger.write_output(
        config.current_project_id, config.current_run_id, output_data.dict()
    )


def _log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    Threaded child function for logging model outputs. Used as target for
    log_model_outputs

    :param outputs: GalileoModelConfig
    :return:
    """
    try:
        outputs.validate()
    except AssertionError as e:
        raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
    for id, prob, emb in zip(outputs.ids, outputs.probs, outputs.emb):
        log_model_output(
            {
                "id": id,
                "epoch": outputs.epoch,
                "split": outputs.split,
                "emb": emb,
                "prob": prob,
                "pred": str(int(np.argmax(prob))),
            }
        )


def log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    First class function to log all model outputs in a training/validation/test
    batch. Use log_model_outputs instead of log_model_outputs to take advantage of
    multithreading and other validation support.

    :param outputs: GalileoModelConfig
    :return: None
    """
    try:
        out_thread = Thread(target=_log_model_outputs, args=[outputs])
        out_thread.start()
    except Exception as e:
        raise GalileoException(e)
