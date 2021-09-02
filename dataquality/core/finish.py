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
    in_out_file_dir = f"{location}/in_out"
    in_out_filepaths = in_out.to_json(filename=in_out_file_dir)

    print("☁️ Uploading Data")
    for io_path in in_out_filepaths:
        fname = os.path.basename(io_path).split(".")[0]
        object_store.create_project_run_object(
            config.current_project_id,
            config.current_run_id,
            object_name=f"in_out/{fname}.jsonl",
            file_path=f"{in_out_file_dir}/{fname}",
        )

    print("☁️ Uploading Embeddings")
    object_store.create_project_run_object(
        config.current_project_id,
        config.current_run_id,
        object_name=JsonlLogger.EMB_LOG_FILENAME,
        file_path=f"{location}/{JsonlLogger.EMB_LOG_FILENAME}",
    )

    print("🧹 Cleaning up")
    shutil.rmtree(location)
