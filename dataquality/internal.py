"""Internal functions to help Galileans"""
import dataquality.metrics
from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType, Route
from dataquality.schemas.task_type import TaskType

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
    task_type = api_client.get_task_type(project, run)

    tasks = []
    ner_labels = []
    try:
        labels = api_client.get_labels_for_run(project_name, run_name)
    except GalileoException as e:
        if "No data found" in str(e):
            e = GalileoException(
                f"It seems no data is available for run " f"{project_name}/{run_name}"
            )
        raise e from None
    # There were no labels available for this run
    except KeyError:
        raise GalileoException(
            "It seems we cannot find the labels for this run. This means "
            "that the run cannot be reprocessed, because it likely was never processed "
            "to begin with. Please call dq.init() and re-train your model "
        ) from None
    # Multi-label has tasks and List[List] for labels
    if task_type == TaskType.text_multi_label:
        tasks = api_client.get_tasks_for_run(project_name, run_name)
    if task_type == TaskType.text_ner:
        # In NER, dq.metrics.get_labels_for_run will return the _full_ label set in NER
        # form (ie B-PER, I-PER, O-PER, B-LOC, etc) which is needed for processing
        ner_labels = dataquality.metrics.get_labels_for_run(project_name, run_name)
    if alerts:
        api_client.delete_alerts(project_name, run_name)
    body = dict(
        project_id=str(project),
        run_id=str(run),
        labels=labels,
        tasks=tasks or None,
        task_type=task_type,
        ner_labels=ner_labels,
        xray=alerts,
        # We set the job name to inference and non_inference_logged to True because
        # This will force the server to first reprocess the non-inference splits,
        # and then reprocess all of the inference splits. If there are no inference
        # splits, this will still work as expected, inference will just be skipped
        job_name="inference",
        process_existing_inference_runs=True,
        non_inference_logged=True,
    )
    res = api_client.make_request(
        RequestType.POST, url=f"{config.api_url}/{Route.jobs}", body=body
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
