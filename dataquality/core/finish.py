import os
import shutil
import threading
from glob import glob
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
import requests
import vaex
from pyarrow.lib import ArrowException, ArrowIOError
from requests import HTTPError

from dataquality import config
from dataquality.clients import object_store
from dataquality.exceptions import GalileoException
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import Pipeline, Route
from dataquality.schemas.split import Split
from dataquality.utils.auth import headers
from dataquality.utils.thread_pool import ThreadPoolManager

lock = threading.Lock()
DATA_FOLDERS = ["emb", "prob", "data"]


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
    prob = in_out[["id", "prob", "gold"]]
    other_cols = [i for i in in_out.columns if i not in ["emb", "prob"]]
    in_out = in_out[other_cols]

    # Each input file will have all records from the same split and epoch, so we can
    # store them in minio partitioned by split/epoch
    epoch, split = in_out[["epoch", "split"]].loc[0]

    # Random name to avoid collisions
    object_name = f"{str(uuid4()).replace('-', '')[:12]}.arrow"
    for file, data_name in zip([emb, prob, in_out], DATA_FOLDERS):
        path = f"{location}/{split}/{epoch}/{data_name}"
        _save_arrow_file(path, object_name, file)


def _save_arrow_file(location: str, file_name: str, file: pd.DataFrame) -> None:
    """
    Helper function to save a pandas dataframe as an arrow file. We use the
    to_feather function as the wrapper to arrow
    https://arrow.apache.org/docs/python/generated/pyarrow.feather.write_feather.html
    Feather is an arrow storage: https://arrow.apache.org/docs/python/feather.html

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


def _upload() -> None:
    """
    Iterates through all of the splits/epochs/[data/emb/prob] folders, concatenates
    all of the files with vaex, and uploads them to a single file in minio in the same
    directory structure
    """
    ThreadPoolManager.wait_for_threads()
    print("☁️ Uploading Data")
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    for split in Split.get_valid_attributes():
        split_loc = f"{location}/{split}"
        if not os.path.exists(split_loc):
            continue
        for epoch_dir in glob(f"{split_loc}/*"):
            for data_folder in DATA_FOLDERS:
                files_dir = f"{epoch_dir}/{data_folder}"
                df = vaex.open(f"{files_dir}/*")
                # Remove the log_file_dir from the object store path
                epoch = epoch_dir.split("/")[-1]
                proj_run = f"{config.current_project_id}/{config.current_run_id}"
                minio_file = (
                    f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.arrow"
                )
                object_store.create_project_run_object_from_df(df, minio_file)


def _cleanup() -> None:
    """
    Cleans up the current run data locally
    """
    assert config.current_project_id
    assert config.current_run_id
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    print("🧹 Cleaning up")
    for path in glob(f"{location}/*"):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def finish() -> Optional[Dict[str, Any]]:
    """
    Finishes the current run and invokes the pipeline to begin processing
    """
    assert config.current_project_id, "You must have an active project to call finish"
    assert config.current_run_id, "You must have an active run to call finish"
    assert config.labels, (
        "You must set your config labels before calling finish. "
        "See `dataquality.set_labels_for_run`"
    )
    assert len(config.labels) == config.observed_num_labels, (
        f"You set your labels to be {config.labels} ({len(config.labels)} labels) "
        f"but based on training, your model "
        f"is expecting {config.observed_num_labels} labels. "
        f"Use dataquality.set_labels_for_run to update your config labels"
    )
    _upload()
    _cleanup()

    # Kick off the default API pipeline to calculate statistics
    # to populate the main home console
    pipeline = Pipeline.default.value

    body = dict(
        project_id=str(config.current_project_id),
        run_id=str(config.current_run_id),
        pipeline_name=pipeline,
        pipeline_env_vars=dict(
            GALILEO_LABELS=config.labels,
        ),
    )
    r = requests.post(
        f"{config.api_url}/{Route.pipelines}",
        json=body,
        headers=headers(config.token),
    )
    try:
        r.raise_for_status()
    except HTTPError:
        try:
            err = "There was an issue with your request. The following was raised:\n"
            details = r.json()["detail"]
            for detail in details:
                err += f'The provided {detail["loc"][-1]} {detail["msg"]}\n'
        except Exception:
            err = (
                f"Your request could not be completed. The following error was "
                f"raised: {r.text}"
            )
        raise GalileoException(err) from None

    res = r.json()
    print(f"Job {res['pipeline_name']} successfully submitted.")
    return res
