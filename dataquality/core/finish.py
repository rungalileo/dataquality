import os
import pickle
import threading
from glob import glob
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
import requests
from requests import HTTPError

from dataquality import config
from dataquality.clients import object_store
from dataquality.exceptions import GalileoException
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import Pipeline, Route, Serialization
from dataquality.utils.auth import headers
from dataquality.utils.thread_pool import ThreadPoolManager

lock = threading.Lock()


def upload(
    cleanup: bool = True, _in_thread: bool = False, _model_output: pd.DataFrame = None
) -> None:
    """
    Uploads the local data to minio and optionally cleans up the local disk

    :param cleanup: Whether to clean up the local disk
    :param _in_thread: Whether this is being called from a threaded process or not
    :param _model_output: The model output to log. If _in_thread is set, _model_output
    must also be set. When uploading from a thread, we will upload the dataset provided
    directly, instead of reading from files as to avoid thread management and locking
    """
    assert config.current_project_id
    assert config.current_run_id

    if _in_thread and _model_output is None:
        raise GalileoException(
            "Threaded uploads require a _model_output. This should"
            "not be used by the end user."
        )
    if not _in_thread:
        ThreadPoolManager.wait_for_threads()

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
    out_frame = (
        pd.read_json(f"{location}/{JsonlLogger.OUTPUT_FILENAME}", lines=True)
        if not _in_thread
        else _model_output
    )
    out_frame = out_frame.astype(dtype=out_frame_dtypes)

    in_out = out_frame.merge(
        in_frame, on=["split", "id", "data_schema_version"], how="left"
    )

    config.observed_num_labels = len(out_frame["prob"].values[0])

    file_type = config.serialization.value
    # Ensure no collisions
    random_name = f"{str(uuid4()).replace('-','')[:12]}"
    object_name = f"{random_name}.{file_type}"
    file_path = f"{location}/{object_name}"
    if config.serialization == Serialization.pickle:
        # Protocol 4 so it is backwards compatible to 3.4 (5 is 3.8)
        records = in_out.to_json(lines=True, orient="records")
        with open(file_path, "wb") as f:
            pickle.dump(records, f, protocol=4)
    else:
        in_out.to_json(file_path, lines=True, orient="records")

    # Causes a lot of prints if we're doing threaded uploading
    if not _in_thread:
        print("☁️ Uploading Data")
    object_store.create_project_run_object(
        config.current_project_id,
        config.current_run_id,
        object_name=f"{object_name}",
        file_path=file_path,
    )

    if cleanup and _in_thread:
        os.remove(file_path)
    elif cleanup:
        _cleanup()


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
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    if len(os.listdir(location)) == 2:  # Have input and model output data
        upload()

    # Kick off the default API pipeline to calculate statistics
    # to populate the main home console
    pipeline = Pipeline.default.value

    body = dict(
        project_id=config.current_project_id,
        run_id=config.current_run_id,
        pipeline_name=pipeline,
        pipeline_env_vars=dict(
            GALILEO_LABELS=config.labels,
            GALILEO_SERIALIZE_MODE=config.serialization.value,
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
    _cleanup()
    return res
