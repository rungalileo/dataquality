import os
import shutil
import threading
from glob import glob
from typing import Any, Dict, Optional

import vaex

from dataquality import config
from dataquality.clients import api_client, object_store
from dataquality.core.log import DATA_FOLDERS
from dataquality.exceptions import GalileoException
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import ProcName, RequestType, Route
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager

lock = threading.Lock()


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

                # Validate all ids within an epoch/split are unique
                if df["id"].nunique() != len(df):
                    epoch = epoch_dir.split("/")[-1]
                    raise GalileoException(
                        "It seems as though you do not have unique ids in this "
                        f"split/epoch. Did you provide your own IDs?\n"
                        f"split:{split}, epoch:{epoch}, ids:{df['id'].tolist()}"
                    )

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
    Finishes the current run and invokes a job to begin processing
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

    body = dict(
        project_id=str(config.current_project_id),
        run_id=str(config.current_run_id),
        proc_name=ProcName.default.value,
        labels=config.labels,
    )
    res = api_client.make_request(
        RequestType.POST, url=f"{config.api_url}/{Route.proc_pool}", body=body
    )
    print(
        f"Job {res['proc_name']} successfully submitted. Results will be available "
        f"soon at {res['link']}"
    )
    return res
