import os
import threading
from typing import Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowException, ArrowIOError
from pydantic.error_wrappers import ValidationError

from dataquality import config
from dataquality.core.integrations.config import GalileoDataConfig, GalileoModelConfig
from dataquality.exceptions import GalileoException
from dataquality.loggers import JsonlLogger
from dataquality.schemas.jsonl_logger import JsonlInputLogItem, JsonlOutputLogItem
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager

DATA_FOLDERS = ["emb", "prob", "data"]


class Logger:
    def __init__(self) -> None:
        self.jsonl_logger = JsonlLogger()


logger = Logger()
lock = threading.Lock()


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


def _log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    Threaded child function for logging model outputs. Used as target for
    log_model_outputs

    :param outputs: GalileoModelConfig
    """
    try:
        outputs.validate()
    except AssertionError as e:
        raise GalileoException(f"The provided GalileoModelConfig is invalid. {e}")
    data = []
    for record_id, prob, emb in zip(outputs.ids, outputs.probs, outputs.emb):
        record = {
            "id": record_id,
            "epoch": outputs.epoch,
            "split": outputs.split,
            "emb": emb,
            "prob": prob,
            "pred": str(int(np.argmax(prob))),
        }
        record = validate_model_output(record).dict()
        data.append(record)
    write_model_output(model_output=pd.DataFrame(data))


def log_model_outputs(outputs: GalileoModelConfig) -> None:
    """
    First class function to log all model outputs in a training/validation/test
    batch. Use log_model_outputs instead of log_model_outputs to take advantage of
    multithreading and other validation support.

    :param outputs: GalileoModelConfig
    """
    ThreadPoolManager.add_thread(target=_log_model_outputs, args=[outputs])


def write_model_output(model_output: pd.DataFrame) -> None:
    """
    Stores a batch log of model output data to disk for batch uploading.

    :param model_output: The model output to log.
    """
    assert config.current_project_id
    assert config.current_run_id

    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    in_frame_dtypes = {"gold": "object"}
    out_frame_dtypes = {"pred": "int64"}
    in_frame = pd.read_json(
        f"{location}/{JsonlLogger.INPUT_FILENAME}",
        lines=True,
        dtype=in_frame_dtypes,
    )
    out_frame = model_output.astype(dtype=out_frame_dtypes)

    in_out = out_frame.merge(
        in_frame, on=["split", "id", "data_schema_version"], how="left"
    )

    config.observed_num_labels = len(out_frame["prob"].values[0])

    # Separate out embeddings and probabilities into their own arrow files
    emb = in_out[["id", "emb"]]
    n_emb = len(emb["emb"][0])
    # We want embeddings to be a wide dataframe 1 column per emb
    cols = [f"emb_{i}" for i in range(n_emb)]
    emb_wide = pd.DataFrame(emb["emb"].tolist(), columns=cols)
    emb_wide.insert(loc=0, column="id", value=emb["id"])

    prob = in_out[["id", "prob", "gold"]]
    other_cols = [i for i in in_out.columns if i not in ["emb", "prob"]]
    in_out = in_out[other_cols]

    # Each input file will have all records from the same split and epoch, so we can
    # store them in minio partitioned by split/epoch
    epoch, split = in_out[["epoch", "split"]].loc[0]

    # Random name to avoid collisions
    object_name = f"{str(uuid4()).replace('-', '')[:12]}.arrow"
    for file, data_name in zip([emb_wide, prob, in_out], DATA_FOLDERS):
        path = f"{location}/{split}/{epoch}/{data_name}"
        _save_arrow_file(path, object_name, file)


def _save_arrow_file(location: str, file_name: str, file: pd.DataFrame) -> None:
    """
    Helper function to save a pandas dataframe as an arrow file. We use the
    to_feather function as the wrapper to arrow.

    We try to save the file with zstd compression first, falling back to default
    (lz4) if zstd is for some reason unavailable. We try zstd first because testing
    has showed better compression levels for our data.
    """
    with lock:
        if not os.path.isdir(location):
            os.makedirs(location)

    file_path = f"{location}/{file_name}"
    try:
        file.to_feather(file_path, compression="zstd")
    # In case zstd compression is not available
    except (ArrowException, ArrowIOError):
        file.to_feather(file_path)
