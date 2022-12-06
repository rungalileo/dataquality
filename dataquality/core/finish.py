import os
from typing import Any, Dict, Optional

from pydantic import UUID4

import dataquality
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.core.report import build_run_report
from dataquality.schemas import RequestType, Route
from dataquality.schemas.job import JobName
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dq_logger import DQ_LOG_FILE_HOME, upload_dq_log_file
from dataquality.utils.helpers import check_noop, open_console_url
from dataquality.utils.thread_pool import ThreadPoolManager
from dataquality.utils.version import _version_check

api_client = ApiClient()
a = Analytics(ApiClient, config)  # type: ignore


@check_noop
def finish(
    last_epoch: Optional[int] = None,
    wait: bool = True,
    create_data_embs: bool = False,
) -> str:
    """
    Finishes the current run and invokes a job

    :param last_epoch: If set, only epochs up to this value will be uploaded/processed
        This is inclusive, so setting last_epoch to 5 would upload epochs 0,1,2,3,4,5
    :param wait: If true, after uploading the data, this will wait for the
        run to be processed by the Galileo server. If false, you can manually wait
        for the run by calling `dq.wait_for_run()` Default True
    :param create_data_embs: If True, an off-the-shelf transformer will run on the raw
        text input to generate data-level embeddings. These will be available in the
        `data view` tab of the Galileo console. You can also access these embeddings
        via dq.metrics.get_data_embeddings()
    """
    a.log_function("dq/finish")
    ThreadPoolManager.wait_for_threads()
    assert config.current_project_id, "You must have an active project to call finish"
    assert config.current_run_id, "You must have an active run to call finish"
    assert config.task_type, "You must have a task type to call finish"
    data_logger = dataquality.get_data_logger()
    data_logger.validate_labels()

    _version_check()

    if data_logger.non_inference_logged():
        _reset_run(config.current_project_id, config.current_run_id, config.task_type)

    data_logger.upload(last_epoch, create_data_embs=create_data_embs)
    upload_dq_log_file()

    body = dict(
        project_id=str(config.current_project_id),
        run_id=str(config.current_run_id),
        labels=data_logger.logger_config.labels,
        task_type=config.task_type.value,
        tasks=data_logger.logger_config.tasks,
        ner_labels=data_logger.logger_config.ner_labels,
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
    if data_logger.logger_config.conditions:
        print(
            "Waiting for run to process before building run report... "
            "Don't close laptop or terminate shell."
        )
        wait_for_run()
        open_console_url(res["link"])
        build_run_report(
            data_logger.logger_config.conditions,
            data_logger.logger_config.report_emails,
            project_id=config.current_project_id,
            run_id=config.current_run_id,
            link=res["link"],
        )
    elif wait:
        wait_for_run()
        open_console_url(res["link"])

    # Reset the environment
    data_logger._cleanup()
    return res.get("link") or ""


@check_noop
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


@check_noop
def get_run_status(
    project_name: Optional[str] = None, run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns the latest job of a specified project run.
    Defaults to the current run if project_name and run_name are empty.
    Raises error if only one of project_name and run_name is passed in.

    :param project_name: The project name. Default to current project if not passed in.
    :param run_name: The run name. Default to current run if not passed in.
    :return: Dict[str, Any]. Response will have key `status` with value
      corresponding to the status of the latest job for the run.
      Other info, such as `created_at`, may be included.
    """
    return api_client.get_run_status(project_name=project_name, run_name=run_name)


@check_noop
def _reset_run(
    project_id: UUID4, run_id: UUID4, task_type: Optional[TaskType] = None
) -> None:
    """Clear the data in minio before uploading new data

    If this is a run that already existed, we want to fully overwrite the old data.
    We can do this by deleting the run and recreating it with the same name, which will
    give it a new ID
    """
    old_run_id = run_id
    api_client.reset_run(project_id, old_run_id, task_type)
    project_dir = (
        f"{dataquality.get_data_logger().LOG_FILE_DIR}/{config.current_project_id}"
    )
    # All of the logged user data is to the old run ID, so rename it to the new ID
    os.rename(f"{project_dir}/{old_run_id}", f"{project_dir}/{config.current_run_id}")
    # Move std logs as well
    os.rename(
        f"{DQ_LOG_FILE_HOME}/{old_run_id}",
        f"{DQ_LOG_FILE_HOME}/{config.current_run_id}",
    )
    config.update_file_config()
