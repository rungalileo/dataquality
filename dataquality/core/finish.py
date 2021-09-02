import os
import shutil

import dask.dataframe as dd

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.jsonl_logger import JsonlLogger


def finish() -> None:
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
        fname = os.path.basename(io_path).split(".")[0]
        object_store.create_project_run_object(
            config.current_project_id,
            config.current_run_id,
            object_name=f"{fname}.jsonl",
            file_path=io_path,
        )

    print("🧹 Cleaning up")
    shutil.rmtree(location)
