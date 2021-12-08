from typing import Any, Dict, Optional

import dataquality
from dataquality import config
from dataquality.clients import api_client
from dataquality.schemas import ProcName, RequestType, Route
from dataquality.utils.version import _version_check


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
    data_logger = dataquality.get_data_logger()
    data_logger.upload()
    data_logger._cleanup()
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
