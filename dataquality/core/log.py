from typing import Dict, List

import numpy as np
import pandas as pd
from pydantic.error_wrappers import ValidationError

import dataquality
from dataquality import config
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.exceptions import GalileoException
from dataquality.loggers import JsonlLogger
from dataquality.schemas.jsonl_logger import JsonlInputLogItem, JsonlOutputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager


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


def log_batch_input_data(data: GalileoDataConfig) -> None:
    """
    First class function to log all input data in batch for a training/validation/test
    batch. Use log_batch_input_data instead of log_input_data to take advantage of
    validation support and logging many records at once

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


def validate_model_output(data: Dict) -> JsonlOutputLogItem:
    """
    Validates the model output data
    """
    try:
        output_data = JsonlOutputLogItem(**data)
    except ValidationError as e:
        raise e
    return output_data


def log_model_output(data: Dict) -> None:
    """
    Function to log a single model output for a train/test/validation dataset.
    Use the log_model_outputs instead to take advantage of threading.

    :param data: Dictionary of model output (id, split, epoch, embeddings,
    probabilities and prediction)
    :return: None
    """
    output_data = validate_model_output(data)

    assert config.current_project_id is not None
    assert config.current_run_id is not None

    logger.jsonl_logger.write_output(
        config.current_project_id, config.current_run_id, output_data.dict()
    )


def set_labels_for_run(labels: List[str]) -> None:
    """
    Creates the mapping of the labels for the model to their respective indexes.

    :param labels: An ordered list of labels (ie ['dog','cat','fish']
    :return: None
    """
    if len(labels) == 1:
        labels = [labels[0], f"NOT_{labels[0]}"]
    config.labels = [str(i) for i in labels]
    config.update_file_config()


def _log_model_outputs(outputs: GalileoModelConfig, upload: bool = True) -> None:
    """
    Threaded child function for logging model outputs. Used as target for
    log_model_outputs

    :param outputs: GalileoModelConfig
    :param upload: Whether or not to immediately upload the logged data
    """
    try:
        outputs.validate()
    except AssertionError as e:
        raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
    data = []
    for id, prob, emb in zip(outputs.ids, outputs.probs, outputs.emb):
        record = {
            "id": id,
            "epoch": outputs.epoch,
            "split": outputs.split,
            "emb": emb,
            "prob": prob,
            "pred": str(int(np.argmax(prob))),
        }
        if upload:
            record = validate_model_output(record).dict()
            data.append(record)
        else:
            log_model_output(record)
    if upload:
        dataquality.upload(_in_thread=True, _model_output=pd.DataFrame(data))


def log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    First class function to log all model outputs in a training/validation/test
    batch. Use log_model_outputs instead of log_model_outputs to take advantage of
    multithreading and other validation support.

    :param outputs: GalileoModelConfig
    """
    try:
        ThreadPoolManager.add_thread(target=_log_model_outputs, args=[outputs, True])
    except Exception as e:
        raise GalileoException(e)
