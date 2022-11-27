from enum import Enum
from typing import Optional, Union

from pydantic import UUID4

from dataquality.schemas.split import Split


class Route(str, Enum):
    """
    List of available API routes
    """

    projects = "projects"
    runs = "runs"
    users = "users"
    cleanup = "cleanup"
    login = "login"
    current_user = "current_user"
    healthcheck = "healthcheck"
    slices = "slices"
    split_path = "split"
    splits = "splits"
    inference_names = "inference_names"
    jobs = "jobs"
    latest_job = "jobs/latest"
    presigned_url = "presigned_url"
    tasks = "tasks"
    labels = "labels"
    epochs = "epochs"
    summary = "insights/summary"
    groupby = "insights/groupby"
    distribution = "insights/distribution"
    xray = "insights/xray"
    export = "export"
    edits = "edits"
    export_edits = "edits/export"
    ampli = "ampli"
    notify = "notify/email"

    @staticmethod
    def content_path(
        project_id: Optional[Union[str, UUID4]] = None,
        run_id: Optional[Union[str, UUID4]] = None,
        split: Optional[Union[str, Split]] = None,
    ) -> str:
        path = ""
        if project_id:
            path += f"/{Route.projects}/{project_id}"
        if run_id:
            path += f"/{Route.runs}/{run_id}"
        if split:
            path += f"/{Route.split_path}/{split}"

        return path.strip("/")
