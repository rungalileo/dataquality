import os
from collections import defaultdict
from typing import Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd
import vaex
from pydantic.error_wrappers import ValidationError
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.exceptions import GalileoException
from dataquality.loggers import JsonlLogger
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.jsonl_logger import JsonlOutputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _save_hdf5_file, _try_concat_df

DATA_FOLDERS = ["emb", "prob", "data"]
INPUT_DATA_NAME = "input_data.arrow"


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


logger = Logger()


def log_batch_input_data(data: GalileoDataConfig) -> None:
    """
    First class function to log all input data for a training/validation/test set

    :param data: GalileoDataConfig
    :return: None
    """
    try:
        data.validate()
    except AssertionError as e:
        raise GalileoException(e) from None

    write_input_dir = (
        f"{logger.jsonl_logger.log_file_dir}/{config.current_project_id}/"
        f"{config.current_run_id}"
    )
    if not os.path.exists(write_input_dir):
        os.makedirs(write_input_dir)
    inp = dict(
        id=data.ids,
        text=data.text,
        split=data.split,
        data_schema_version=__data_schema_version__,
        gold=data.labels if data.split != Split.inference.value else None,
        **data.meta
    )
    df = vaex.from_pandas(pd.DataFrame(inp))
    file_path = f"{write_input_dir}/{INPUT_DATA_NAME}"
    if os.path.isfile(file_path):
        new_name = f"{write_input_dir}/{str(uuid4()).replace('-', '')[:12]}.arrow"
        os.rename(file_path, new_name)
        vaex.concat([df, vaex.open(new_name)]).export(file_path)
        os.remove(new_name)
    else:
        df.export(file_path)
    df.close()


def validate_model_output(data: Dict) -> JsonlOutputLogItem:
    """
    Validates the model output data
    """
    try:
        output_data = JsonlOutputLogItem(**data)
    except ValidationError as e:
        raise e
    return output_data


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


def _log_model_outputs_target(outputs: GalileoModelConfig) -> None:
    """
    Threaded child function for logging model outputs. Used as target for
    log_model_outputs

    :param outputs: GalileoModelConfig
    """
    try:
        outputs.validate()
    except AssertionError as e:
        raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
    data = defaultdict(list)
    for record_id, prob, emb in zip(outputs.ids, outputs.probs, outputs.emb):
        record = {
            "id": record_id,
            "epoch": outputs.epoch,
            "split": outputs.split,
            "emb": emb,
            "prob": prob,
            "pred": int(np.argmax(prob)),
        }
        record = validate_model_output(record).dict()
        for k in record.keys():
            data[k].append(record[k])
    write_model_output(model_output=vaex.from_dict(data))


def _log_model_outputs(outputs: GalileoModelConfig) -> None:
    try:
        _log_model_outputs_target(outputs)
    except Exception as e:
        print(f"An error occurred while logging: {str(e)}")
        import traceback

        traceback.print_exc()


def log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    First class function to log all model outputs in a training/validation/test
    batch. Use log_model_outputs instead of log_model_outputs to take advantage of
    multithreading and other validation support.

    :param outputs: GalileoModelConfig
    """
    ThreadPoolManager.add_thread(target=_log_model_outputs, args=[outputs])


def write_model_output(model_output: DataFrame) -> None:
    """
    Stores a batch log of model output data to disk for batch uploading.

    :param model_output: The model output vaex df to log.
    """
    assert config.current_project_id
    assert config.current_run_id

    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )

    config.observed_num_labels = len(model_output["prob"].values[0])
    epoch, split = model_output[["epoch", "split"]][0]
    path = f"{location}/{split}/{epoch}"
    object_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"

    _save_hdf5_file(path, object_name, model_output)
    _try_concat_df(path)
