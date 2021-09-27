import os
import pickle
import shutil
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


def upload(cleanup: bool = True) -> None:
    """
    Uploads the local data to minio and optionally cleans up the local disk

    :param cleanup: Whether to clean up the local disk
    :return:
    """
    assert config.current_project_id
    assert config.current_run_id
    ThreadPoolManager.wait_for_threads()
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    in_frame = pd.read_json(f"{location}/{JsonlLogger.INPUT_FILENAME}", lines=True)
    out_frame = pd.read_json(f"{location}/{JsonlLogger.OUTPUT_FILENAME}", lines=True)
    in_out = in_frame.merge(
        out_frame, on=["split", "id", "data_schema_version"], how="left"
    )

    config.observed_num_labels = len(out_frame["prob"].values[0])

    file_type = config.serialization.value
    object_name = f"{str(uuid4())[:7]}.{file_type}"
    file_path = f"{location}/{object_name}"
    if config.serialization == Serialization.pickle:
        # Protocol 4 so it is backwards compatible to 3.4 (5 is 3.8)
        records = in_out.to_json(lines=True, orient="records")
        with open(file_path, "wb") as f:
            pickle.dump(records, f, protocol=4)
    else:
        in_out.to_json(file_path, lines=True, orient="records")

    print("☁️ Uploading Data")
    object_store.create_project_run_object(
        config.current_project_id,
        config.current_run_id,
        object_name=f"{object_name}",
        file_path=file_path,
    )

    if cleanup:
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
    shutil.rmtree(location)


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
    if os.path.exists(location):
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
    return res
