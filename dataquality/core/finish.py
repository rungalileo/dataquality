import shutil

import pandas as pd

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.jsonl_logger import JsonlLogger


def finish(cleanup: bool = True) -> None:
    assert config.current_project_id
    assert config.current_run_id
    location = (
        f"{JsonlLogger.LOG_FILE_DIR}/{config.current_project_id}"
        f"/{config.current_run_id}"
    )
    in_frame = pd.read_json(f"{location}/{JsonlLogger.INPUT_FILENAME}", lines=True)
    out_frame = pd.read_json(f"{location}/{JsonlLogger.OUTPUT_FILENAME}", lines=True)
    in_out = in_frame.merge(
        out_frame, on=["split", "id", "data_schema_version"], how="left"
    )
    # Protocol 4 so it is backwards compatible to 3.4 (5 is 3.8)
    in_out.to_pickle(f"{location}/file.pkl", protocol=4)

    print("☁️ Uploading Data")
    object_store.create_project_run_object(
        config.current_project_id,
        config.current_run_id,
        object_name="file.pkl",
        file_path=f"{location}/file.pkl",
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
