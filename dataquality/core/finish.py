from typing import Any, Dict, Optional

import dataquality
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.schemas import RequestType, Route
from dataquality.schemas.job import JobName
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.version import _version_check

api_client = ApiClient()


def finish() -> Optional[Dict[str, Any]]:
    """
    Finishes the current run and invokes a job
    """
    ThreadPoolManager.wait_for_threads()
    assert config.current_project_id, "You must have an active project to call finish"
    assert config.current_run_id, "You must have an active run to call finish"
    data_logger = dataquality.get_data_logger()
    data_logger.validate_labels()

    _version_check()

    if data_logger.non_inference_logged():
        # Clear the data in minio before uploading new data
        # If this is a run that already existed, we want to fully overwrite the old data
        # If only inference is logged, keep all existing minio data
        api_client.reset_run(config.current_project_id, config.current_run_id)

    data_logger.upload()
    data_logger._cleanup()

    body = dict(
        project_id=str(config.current_project_id),
        run_id=str(config.current_run_id),
        labels=data_logger.logger_config.labels,
        tasks=data_logger.logger_config.tasks,
    )
    if data_logger.logger_config.inference_logged:
        body.update(
            job_name=JobName.inference,
            non_inference_logged=data_logger.non_inference_logged(),
        )
    res = api_client.make_request(
        RequestType.POST, url=f"{config.api_url}/{Route.jobs}", body=body
    )
    print(
        f"Job {res['job_name']} successfully submitted. Results will be available "
        f"soon at {res['link']}"
    )
    # Reset all config variables
    data_logger.logger_config.reset()
    return res


def wait_for_run(
    project_name: Optional[str] = None, run_name: Optional[str] = None
) -> None:
    """
    Waits until a specific project run transitions from started to finished.
    Defaults to the current run if project_name and run_name are empty.
    Raises error if only one of project_name and run_name is passed in.

    :param project_name: The project name. Default to current project if not passed in.
    :param run_name: The run name. Default to current run if not passed in.
    :return: None. Function returns after the run transitions to `finished`
    """
    return api_client.wait_for_run(project_name=project_name, run_name=run_name)


def get_run_status(
    project_name: Optional[str] = None, run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns the status of a specified project run.
    Defaults to the current run if project_name and run_name are empty.
    Raises error if only one of project_name and run_name is passed in.

    :param project_name: The project name. Default to current project if not passed in.
    :param run_name: The run name. Default to current run if not passed in.
    :return: Dict[str, Any]. Response will have key `status` with value corresponding
      to the status of the run. Other info, such as `timestamp`, may be included.
    """
    return api_client.get_run_status(project_name=project_name, run_name=run_name)
