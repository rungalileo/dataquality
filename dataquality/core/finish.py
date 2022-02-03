from typing import Any, Dict, Optional

import dataquality
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.schemas import ProcName, RequestType, Route
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.version import _version_check

api_client = ApiClient()


def finish() -> Optional[Dict[str, Any]]:
    """
    Finishes the current run and invokes a job to begin processing
    """
    ThreadPoolManager.wait_for_threads()
    assert config.current_project_id, "You must have an active project to call finish"
    assert config.current_run_id, "You must have an active run to call finish"
    data_logger = dataquality.get_data_logger()
    data_logger.validate_labels()

    _version_check()
    # Clear the data in minio before uploading new data
    # If this is a run that already existed, we want to fully overwrite the old data
    api_client.reset_run(config.current_project_id, config.current_run_id)

    data_logger.upload()
    data_logger._cleanup()
    config.update_file_config()

    body = dict(
        project_id=str(config.current_project_id),
        run_id=str(config.current_run_id),
        proc_name=ProcName.default.value,
        labels=data_logger.logger_config.labels,
        tasks=data_logger.logger_config.tasks,
    )
    res = api_client.make_request(
        RequestType.POST, url=f"{config.api_url}/{Route.proc_pool}", body=body
    )
    print(
        f"Job {res['proc_name']} successfully submitted. Results will be available "
        f"soon at {res['link']}"
    )
    return res
