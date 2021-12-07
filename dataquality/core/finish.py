import os
import shutil
from glob import glob
from typing import Any, Dict, Optional

import vaex

from dataquality import config
from dataquality.clients import api_client, object_store
from dataquality.core.log import DATA_FOLDERS, INPUT_DATA_NAME
from dataquality.loggers.jsonl_logger import JsonlLogger
from dataquality.schemas import ProcName, RequestType, Route
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.vaex import _join_in_out_frames, _validate_unique_ids
from dataquality.utils.version import _version_check


def _upload() -> None:
    """
    Iterates through all of the splits/epochs/[data/emb/prob] folders, concatenates
    all of the files with vaex, and uploads them to a single file in minio in the same
    directory structure
    """
    ThreadPoolManager.wait_for_threads()
    print("☁️ Uploading Data")
    proj_run = f"{config.current_project_id}/{config.current_run_id}"
    location = f"{JsonlLogger.LOG_FILE_DIR}/{proj_run}"

    in_frame = vaex.open(f"{location}/{INPUT_DATA_NAME}").copy()
    for split in Split.get_valid_attributes():
        split_loc = f"{location}/{split}"
        if not os.path.exists(split_loc):
            continue
        for epoch_dir in glob(f"{split_loc}/*"):
            epoch = int(epoch_dir.split("/")[-1])

            out_frame = vaex.open(f"{epoch_dir}/*")
            _validate_unique_ids(out_frame)
            in_out = _join_in_out_frames(in_frame, out_frame)

            in_out["id"] = in_out["id"].astype("int64")

            # Separate out embeddings and probabilities into their own files
            prob = in_out[["id", "prob", "gold"]]
            emb = in_out[["id", "emb"]]
            ignore_cols = ["emb", "prob", "gold", "split_id"]
            other_cols = [i for i in in_out.get_column_names() if i not in ignore_cols]
            in_out = in_out[other_cols]

            for data_folder, df_obj in zip(DATA_FOLDERS, [emb, prob, in_out]):
                minio_file = (
                    f"{proj_run}/{split}/{epoch}/{data_folder}/{data_folder}.hdf5"
                )
                object_store.create_project_run_object_from_df(df_obj, minio_file)


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
    _version_check()
    # Clear the data in minio before uploading new data
    # If this is a run that already existed, we want to fully overwrite the old data
    api_client.reset_run(config.current_project_id, config.current_run_id)
    _upload()
    _cleanup()
    config.update_file_config()

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
