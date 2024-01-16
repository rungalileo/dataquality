"""Internal functions to help Galileans"""
from typing import Dict

from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType, Route

api_client = ApiClient()


def reprocess_run(
    project_name: str, run_name: str, alerts: bool = True, wait: bool = True
) -> None:
    """Reprocesses a run that has already been processed by Galileo

    Useful if a new feature has been added to the system that is desired to be added
    to an old run that hasn't been migrated

    :param project_name: The name of the project
    :param run_name: The name of the run
    :param alerts: Whether to create the alerts. If True, all alerts for the run will
        be removed, and recreated during processing. Default True
    :param wait: Whether to wait for the run to complete processing on the server. If
        True, this will block execution, printing out the status updates of the run.
        Useful if you want to know exactly when your run completes. Otherwise, this will
        fire and forget your process. Default True
    """
    project, run = api_client._get_project_run_id(project_name, run_name)
    api_client.get_task_type(project, run)

    job = api_client.get_run_status(project_name, run_name)
    if not job:
        raise GalileoException(
            "It seems we cannot find the job for this run. This means "
            "that the run cannot be reprocessed, because it likely was never processed "
            "to begin with. Please call dq.init() and re-train your model "
        ) from None

    if alerts:
        api_client.delete_alerts(project_name, run_name)

    job_data: Dict = job["request_data"]
    # We need to remove the job_id from the job_data since the server will
    # generate a new one
    job_data.pop("job_id")
    res = api_client.make_request(
        RequestType.POST, url=f"{config.api_url}/{Route.jobs}", body=job_data
    )
    print(
        f"Job {res['job_name']} successfully resubmitted. New results will be "
        f"available soon at {res['link']}"
    )
    if wait:
        api_client.wait_for_run(project_name, run_name)
    return res


def rename_run(project_name: str, run_name: str, new_name: str) -> None:
    """Assigns a new name to a run

    Useful if a run was named incorrectly, or if a run was created with a temporary
    name and needs to be renamed to something more permanent

    :param project_name: The name of the project
    :param run_name: The name of the run
    :param new_name: The new name to assign to the run
    """
    api_client.update_run_name(project_name, run_name, new_name)
    print(
        f"Successfully renamed run {project_name}/{run_name} to "
        f"{project_name}/{new_name}"
    )


def rename_project(project_name: str, new_name: str) -> None:
    """Renames a project

    Useful if a project was named incorrectly, or if a project was created with a
    temporary name and needs to be renamed to something more permanent

    :param project_name: The name of the project
    :param new_name: The new name to assign to the project
    """
    api_client.update_project_name(project_name, new_name)
    print(f"Successfully renamed project {project_name} to " f"{new_name}")
