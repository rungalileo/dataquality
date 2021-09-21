import os
import shutil
from uuid import uuid4

import dask.dataframe as dd
import requests

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import Pipeline, Route
from dataquality.utils.auth import headers


def upload(cleanup: bool = True) -> None:
    """
    Uploads the local data to minio and optionally cleans up the local disk

    :param cleanup: Whether to clean up the local disk
    :return:
    """
    assert config.current_project_id
    assert config.current_run_id
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    in_frame = dd.read_json(f"{location}/{JsonlLogger.INPUT_FILENAME}", lines=True)
    out_frame = dd.read_json(f"{location}/{JsonlLogger.OUTPUT_FILENAME}", lines=True)
    in_out = in_frame.merge(out_frame, on=["split", "id"], how="left")
    in_out_filepaths = in_out.to_json(filename=location)

    print("☁️ Uploading Data")
    for io_path in in_out_filepaths:
        object_store.create_project_run_object(
            config.current_project_id,
            config.current_run_id,
            object_name=f"{str(uuid4())[:7]}.jsonl",
            file_path=io_path,
        )

    if cleanup:
        print("🧹 Cleaning up")
        shutil.rmtree(location)


def cleanup() -> None:
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


def finish() -> None:
    """
    Finishes the current run and invokes the pipeline to begin processing
    """
    assert config.current_project_id, "You must have an active project to call finish"
    assert config.current_run_id, "You must have an active run to call finish"
    assert config.labels, (
        "You must set your config labels before calling finish. "
        "See `dataquality.set_labels_for_run`"
    )
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    if os.path.exists(location):
        upload()

    # Kick off API pipeline to calculate statistics
    r = requests.post(
        f"{config.api_url}/{Route.pipelines}",
        json=dict(
            project_id=str(config.current_project_id),
            run_id=str(config.current_run_id),
            pipeline_name=Pipeline.calculate_metrics,
            pipeline_env_vars=dict(labels=str(config.labels)),
        ),
        headers=headers(config.token),
    )
    r.raise_for_status()
    return r.json()
