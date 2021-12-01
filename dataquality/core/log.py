import os
import threading
from glob import glob
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

DATA_FOLDERS = ["emb", "prob", "data"]
INPUT_DATA_NAME = "input_data.arrow"


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
    input_data = data
    input_data["data_schema_version"] = __data_schema_version__
    assert config.current_project_id is not None
    assert config.current_run_id is not None
    logger.jsonl_logger.write_input(
        config.current_project_id, config.current_run_id, input_data
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

    write_input_dir = (
        f"{logger.jsonl_logger.log_file_dir}/{config.current_project_id}/"
        f"{config.current_run_id}"
    )
    if not os.path.exists(write_input_dir):
        os.makedirs(write_input_dir)
    df = vaex.from_pandas(
        pd.DataFrame(
            dict(
                id=data.ids,
                text=data.text,
                split=data.split,
                data_schema_version=1,
                gold=data.labels if data.split != Split.validation else None,
            )
        )
    )
    file_path = f"{write_input_dir}/{INPUT_DATA_NAME}"
    if os.path.isfile(file_path):
        df = vaex.concat([df, vaex.open(file_path)])
    df.export_arrow(file_path)
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


def _log_model_outputs(outputs: GalileoModelConfig) -> None:
    try:
        _log_model_outputs_target(outputs)
    except Exception as e:
        print(f"An error occurred while logging: {e}")


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

    config.observed_num_labels = len(model_output["prob"].values[0])
    n_emb = len(model_output["emb"][0])
    # We want embeddings to be a wide dataframe 1 column per emb
    emb_cols = [f"emb_{i}" for i in range(n_emb)]
    model_output[emb_cols] = pd.DataFrame(
        model_output["emb"].tolist(), columns=emb_cols
    )

    out_frame_dtypes = {"pred": "int64"}
    in_frame = vaex.open(f"{location}/{INPUT_DATA_NAME}").copy()

    out_frame = vaex.from_pandas(model_output.astype(dtype=out_frame_dtypes))

    in_frame["split_id"] = in_frame["split"] + in_frame["id"].astype("string")
    out_frame["split_id"] = out_frame["split"] + out_frame["id"].astype("string")

    in_out = out_frame.join(
        in_frame, on="split_id", how="left", lsuffix="_L", rsuffix="_R"
    ).copy()
    drop_cols = [c for c in in_out.get_column_names() if not c.endswith("_L")]
    in_out = in_out[drop_cols]
    for c in in_out.get_column_names():
        if c.endswith("_R"):
            in_out.rename(c, c.rstrip("_R"))

    # Separate out embeddings and probabilities into their own arrow files
    prob = in_out[["id", "prob", "gold"]]
    emb_wide = in_out[["id"] + emb_cols]
    ignore_cols = emb_cols + ["emb", "prob", "split_id"]
    other_cols = [i for i in in_out.columns if i not in ignore_cols]
    in_out = in_out[other_cols]

    # Each input file will have all records from the same split and epoch, so we can
    # store them in minio partitioned by split/epoch
    epoch, split = in_out[["epoch", "split"]][0]

    # Random name to avoid collisions
    object_name = f"{str(uuid4()).replace('-', '')[:12]}.arrow"
    for file, data_name in zip([emb_wide, prob, in_out], DATA_FOLDERS):
        path = f"{location}/{split}/{epoch}/{data_name}"
        _save_arrow_file(path, object_name, file)
        file.close()


def _save_arrow_file(location: str, file_name: str, file: DataFrame) -> None:
    """
    Helper function to save a vaex dataframe as an arrow file. We use the
    to_feather function as the wrapper to arrow.

    We try to save the file with zstd compression first, falling back to default
    (lz4) if zstd is for some reason unavailable. We try zstd first because testing
    has showed better compression levels for our data.
    """
    with lock:
        if not os.path.isdir(location):
            os.makedirs(location)
        file_path = f"{location}/{file_name}"
        file.export_arrow(file_path)
        new_name = f"{str(uuid4()).replace('-', '')[:12]}.arrow"
        arrow_files = glob(f"{location}/*.arrow")
        if len(arrow_files) > 25:
            df = vaex.open_many(arrow_files)
            df.export_arrow(f"{location}/{new_name}")
            df.close()
            for f in arrow_files:
                os.remove(f)
