import os

from dataquality import config
from dataquality.clients import object_store
from dataquality.loggers.jsonl_logger import JsonlLogger


def finish() -> None:
    # TODO: build final frame and upload!
    # TODO: will need to upload embeddings and joined input and output
    assert config.current_project_id
    assert config.current_run_id
    for fname in [
        JsonlLogger.INPUT_FILENAME,
        JsonlLogger.OUTPUT_FILENAME,
        JsonlLogger.EMB_LOG_FILENAME,
    ]:
        print(f"☁️ Uploading {fname}")
        object_store.create_project_run_object(
            config.current_project_id,
            config.current_run_id,
            object_name=fname,
            file_path=f"{os.getcwd()}/.galileo/logs/{config.current_project_id}"
            f"/{config.current_run_id}/{fname}",
        )
