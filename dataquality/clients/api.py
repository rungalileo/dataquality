import json
import os
from json import JSONDecodeError
from time import sleep, time
from typing import Any, Dict, List, Optional, Tuple, Union

import jwt
import requests
from pydantic.types import UUID4
from requests import Response

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType, Route
from dataquality.schemas.dataframe import FileType
from dataquality.schemas.edit import Edit
from dataquality.schemas.ner import TaggingSchema
from dataquality.schemas.split import conform_split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.auth import headers


class ApiClient:
    def _refresh_jwt_token(self) -> str:
        username = os.getenv("GALILEO_USERNAME")
        password = os.getenv("GALILEO_PASSWORD")
        api_key = os.getenv("GALILEO_API_KEY")

        if api_key is not None:
            res = requests.post(
                f"{config.api_url}/login/api_key",
                json={"api_key": api_key},
            )
        elif username is not None and password is not None:
            res = requests.post(
                f"{config.api_url}/login",
                data={
                    "username": username,
                    "password": password,
                    "auth_method": "email",
                },
            )
        else:
            raise GalileoException(
                "You are not logged in. Call dataquality.login()\n"
                "GALILEO_USERNAME and GALILEO_PASSWORD must be set"
            )

        if res.status_code != 200:
            raise GalileoException(
                (
                    f"Issue authenticating: {res.json()['detail']} "
                    "If you need to reset your password, "
                    f"go to {config.api_url.replace('api', 'console')}/forgot-password"
                )
            )

        access_token = res.json().get("access_token", "")
        config.token = access_token
        config.update_file_config()
        return access_token

    def get_token(self) -> str:
        token = config.token
        if not token:
            token = self._refresh_jwt_token()

        # Check to see if our token is expired before making a request
        # and refresh token if it's expired
        # if url is not Routes.login and self.token:
        if token:
            claims = jwt.decode(token, options={"verify_signature": False})
            if claims.get("exp", 0) < time():
                token = self._refresh_jwt_token()

        return token

    def make_request(
        self,
        request: RequestType,
        url: str,
        body: Optional[Dict] = None,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        header: Optional[Dict] = None,
        timeout: Union[int, None] = None,
        files: Optional[Dict] = None,
        return_response_without_validation: bool = False,
    ) -> Any:
        """Makes an HTTP request.

        This is the center point of all functions and the main entry/exit for the
        dataquality client to interact with the server.
        """
        token = self.get_token()
        header = header or headers(token)
        res = RequestType.get_method(request.value)(
            url,
            json=body,
            params=params,
            headers=header,
            data=data,
            timeout=timeout,
            files=files,
        )
        if return_response_without_validation:
            return res
        self._validate_response(res)
        return res.json()

    def _validate_response(self, res: Response) -> None:
        if not res.ok:
            msg = (
                "Something didn't go quite right. The api returned a non-ok status "
                f"code {res.status_code} with output: {res.text}"
            )
            raise GalileoException(msg)
        # If a run/split or a set of filters has no data, the API will return a 200
        # with a special header letting us know, so we don't try to load a file
        # with an unexpected format
        elif res.headers.get("Galileo-No-Data") == "true":
            msg = (
                "It seems there is no data for this request.\nEnsure you spelled "
                "everything correctly. Projects and runs are case sensitive."
            )
            raise GalileoException(msg)

    def _get_user_id(self) -> UUID4:
        return self.get_current_user()["id"]

    def get_current_user(self) -> Dict:
        if not self.get_token():
            raise GalileoException(
                "Current user is not set! Please ensure GALILEO_USERNAME "
                "is set to the registered email"
            )

        return self.make_request(
            RequestType.GET, url=f"{config.api_url}/{Route.current_user}"
        )

    def valid_current_user(self) -> bool:
        try:
            self.get_current_user()
            return True
        except GalileoException:
            return False

    def get_project(self, project_id: UUID4) -> Dict:
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{project_id}",
        )

    def get_projects(self) -> List[Dict]:
        user_id = self._get_user_id()
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.users}/{user_id}/{Route.projects}",
        )

    def get_project_by_name(self, project_name: str) -> Dict:
        projs = self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}",
            params={"project_name": project_name},
        )
        return projs[0] if projs else {}

    def get_project_runs(self, project_id: UUID4) -> List[Dict]:
        """Gets all runs from a project by ID"""
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}",
        )

    def get_project_runs_by_name(self, project_name: str) -> List[Dict]:
        """Gets all runs from a project by name"""
        proj = self.get_project_by_name(project_name)
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}",
        )

    def get_project_run(self, project_id: UUID4, run_id: UUID4) -> Dict:
        """Gets a run in a project by ID"""
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}",
        )

    def get_project_run_by_name(self, project_name: str, run_name: str) -> Dict:
        proj = self.get_project_by_name(project_name)
        if not proj:
            raise GalileoException(f"No project with name {project_name}")
        url = f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}"
        params = {"run_name": run_name}
        runs = self.make_request(RequestType.GET, url=url, params=params)
        return runs[0] if runs else {}

    def update_run_name(self, project_name: str, run_name: str, new_name: str) -> Dict:
        project_id, run_id = self._get_project_run_id(project_name, run_name)
        if not project_id:
            raise GalileoException(f"No project with name {project_name}")
        if not run_id:
            raise GalileoException(f"No run with name {run_name}")

        url = f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}"
        data = {"name": new_name}
        run = self.make_request(RequestType.PUT, url=url, body=data)
        return run if run else {}

    def update_project_name(self, project_name: str, new_name: str) -> Dict:
        proj = self.get_project_by_name(project_name)
        if not proj:
            raise GalileoException(f"No project with name {project_name}")

        new_proj = self.get_project_by_name(new_name)
        if new_proj:
            raise GalileoException(
                f"You cannot rename to an existing project name {new_name}"
            )

        url = f"{config.api_url}/{Route.projects}/{proj['id']}"
        data = {"name": new_name}
        proj_resp = self.make_request(RequestType.PUT, url=url, body=data)
        return proj_resp if proj_resp else {}

    def create_project(self, project_name: str) -> Dict:
        """Creates a project given a name and returns the project information"""
        body = {"name": project_name, "type": "training_inference"}
        return self.make_request(
            RequestType.POST, url=f"{config.api_url}/{Route.projects}", body=body
        )

    def create_run(self, project_name: str, run_name: str, task_type: TaskType) -> Dict:
        """Creates a run in a given project"""
        body = {"name": run_name, "task_type": task_type.value}
        proj = self.get_project_by_name(project_name)
        return self.make_request(
            RequestType.POST,
            url=f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}",
            body=body,
        )

    def reset_run(
        self, project_id: UUID4, run_id: UUID4, task_type: Optional[TaskType] = None
    ) -> None:
        """Resets a run by deleting the run with that name and creating a new one
        with the same name, getting a new UUID

        Called before any call to `dataquality.finish` if prior data was logged.
        see `dataquality.finish`
        """
        project_name = self.get_project(project_id)["name"]
        run = self.get_project_run(project_id, run_id)
        run_name = run["name"]
        task_type = task_type or TaskType.get_mapping(run["task_type"])

        # Delete the run
        self.delete_run(project_id, run_id)
        # Create a run with the same name
        new_run = self.create_run(project_name, run_name, task_type)
        # Update config
        config.current_run_id = new_run["id"]

    def delete_run(self, project_id: UUID4, run_id: UUID4) -> Dict:
        """Deletes a run

        This clears all metadata about the run, all object data, and the run itself
        """
        return self.make_request(
            RequestType.DELETE,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}",
        )

    def delete_run_by_name(self, project_name: str, run_name: str) -> None:
        """Deletes a run via name

        This clears all metadata about the run, all object data, and the run itself
        """
        run = self.get_project_run_by_name(project_name, run_name)
        if not run:
            raise GalileoException(
                f"No project/run found with name {project_name}/{run_name}"
            )
        project_id = run["project_id"]
        run_id = run["id"]
        return self.make_request(
            RequestType.DELETE,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}",
        )

    def delete_project(self, project_id: UUID4) -> Dict:
        """Deletes a project

        For each run in the project, this clears all metadata about the run,
        all object data, and the run itself
        """
        runs = self.get_project_runs(project_id)
        print("Deleting all runs within project.")
        for run in runs:
            print(f"Deleting run {run['name']}", end="... ")
            self.delete_run(project_id, run["id"])
            print("Done.")
        return self.make_request(
            RequestType.DELETE,
            url=f"{config.api_url}/{Route.projects}/{project_id}",
        )

    def delete_project_by_name(self, project_name: str) -> None:
        """Deletes a project by name

        For each run in the project, this clears all metadata about the run,
        all object data, and the run itself
        """
        project = self.get_project_by_name(project_name)
        if not project:
            raise GalileoException(f"No project found with name {project_name}")
        self.delete_project(project["id"])

    def _get_project_run_id(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> Tuple[UUID4, UUID4]:
        """Helper function to get the project/run ids

        If project and run names are provided, the IDs will be fetched. Otherwise,
        This will return the currently initialized project/run IDs
        """
        if (project_name and not run_name) or (run_name and not project_name):
            raise GalileoException(
                "You must either provide both a project and run"
                "name or neither. If you provide neither, the "
                "currently initialized project/run will be used."
            )
        elif project_name and run_name:
            res = self.get_project_run_by_name(project_name, run_name)
            if not res:
                raise GalileoException(
                    f"No project/run found with name {project_name}/{run_name}"
                )
            project = res["project_id"]
            run = res["id"]
        else:
            project = config.current_project_id
            run = config.current_run_id
        return project, run

    def get_labels_for_run(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        task: Optional[str] = None,
    ) -> List[str]:
        """Gets the labels for a given run, else the currently initialized project/run

        If you do not provide a project and run name, the currently initialized
        project/run will be used. Otherwise you must provide both a project and run name
        If the run is a multi-label run, a task must be provided
        """
        project, run = self._get_project_run_id(
            project_name=project_name, run_name=run_name
        )
        url = f"{config.api_url}/{Route.content_path(project, run)}/{Route.labels}"
        params = {"task": task} if task else None
        res = self.make_request(RequestType.GET, url=url, params=params)
        return res["labels"]

    def get_tasks_for_run(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> List[str]:
        """Gets the task names for a given multi-label run,

        If you do not provide a project and run name, the currently initialized
        project/run will be used. Otherwise you must provide both a project and run name

        This function is only valid for multi-label runs.
        """
        project, run = self._get_project_run_id(
            project_name=project_name, run_name=run_name
        )
        if self.get_task_type(project, run) != TaskType.text_multi_label:
            return []
        url = f"{config.api_url}/{Route.content_path(project, run)}/{Route.tasks}"
        res = self.make_request(RequestType.GET, url=url)
        return res["tasks"]

    def get_epochs_for_run(
        self, project_name: str, run_name: str, split: str
    ) -> List[int]:
        """Returns an ordered list of epochs for a run"""
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)
        url = (
            f"{config.api_url}/{Route.content_path(project, run, split)}/{Route.epochs}"
        )
        return self.make_request(RequestType.GET, url=url)

    def create_edit(self, edit: Edit) -> Dict:
        assert edit.project_id and edit.run_id and edit.split
        split = conform_split(edit.split)
        path = Route.content_path(edit.project_id, edit.run_id, split)
        url = f"{config.api_url}/{path}/{Route.edits}"
        body = edit.dict()
        params = {"inference_name": edit.inference_name}
        return self.make_request(RequestType.POST, url=url, body=body, params=params)

    def reprocess_run(
        self,
        project_name: Optional[str] = None,
        run_name: Optional[str] = None,
        labels: Optional[Union[List, List[List]]] = None,
        xray: bool = False,
    ) -> Dict:
        """Removed. Please see dq.internal.reprocess_run"""
        raise GalileoException(
            "It seems you are trying to reprocess a run. Please call the following:\n\n"
            "from dataquality.internal import reprocess_run\n\n"
            f"reprocess_run('{project_name}', '{run_name}')"
        )

    def get_slice_by_name(self, project_name: str, slice_name: str) -> Dict:
        """Get a slice by name"""
        proj = self.get_project_by_name(project_name)
        url = f"{config.api_url}/{Route.content_path(proj['id'])}/{Route.slices}"
        params = {"slice_name": slice_name}
        slices = self.make_request(RequestType.GET, url=url, params=params)
        if not slices:
            raise GalileoException(
                f"No slice found for project {project_name} with name {slice_name}"
            )
        return slices[0]

    def get_metadata_columns(
        self, project_name: str, run_name: str, split: str
    ) -> Dict:
        """Lists the available metadata columns for a run/split

        Structure of data is:
        [{
            "name": str
            "is_categorical": bool
            "unique_values": Optional[List]
            "max": Optional[float]
            "min": Optional[float]
        },...]
        :param project_name:
        :param run_name:
        :param split:
        """
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)
        url = f"{config.api_url}/{Route.content_path(project, run, split)}/meta/columns"
        return self.make_request(RequestType.POST, url, body={})

    def get_task_type(self, project_id: UUID4, run_id: UUID4) -> TaskType:
        return TaskType.get_mapping(
            self.get_project_run(project_id, run_id)["task_type"]
        )

    def export_run(
        self,
        project_name: str,
        run_name: str,
        split: str,
        file_name: str,
        inference_name: str = "",
        slice_name: Optional[str] = None,
        include_cols: Optional[List[str]] = None,
        col_mapping: Optional[Dict[str, str]] = None,
        hf_format: bool = False,
        tagging_schema: Optional[TaggingSchema] = None,
        filter_params: Optional[Dict] = None,
    ) -> None:
        """Export a project/run to disk as a file

        :param project_name: The project name
        :param run_name: The run name
        :param split: The split to export on
        :param file_name: The file name. Must end in a supported FileType
        :param inference_name: Required if split is inference. The name of the inference
            split to get data for.
        :param slice_name: The optional slice name to export. If selected, this data
        from this slice will be exported only.
        :param include_cols: List of columns to include in the export. If not set,
        all columns will be exported. If "*" is included, return all metadata columns
        :param col_mapping: Dictionary of renamed column names for export.
        :param hf_format: (NER only)
            Whether to export the dataframe in a HuggingFace compatible format
        :param tagging_schema: (NER only)
            If hf_format is True, you must pass a tagging schema
        :param filter_params: Filters to apply to the dataframe before exporting. Only
            rows with matching filters will be included in the exported data
        """
        project, run = self._get_project_run_id(project_name, run_name)
        ext = os.path.splitext(file_name)[-1].lstrip(".")

        assert ext in list(FileType), f"File must be one of {list(FileType)}"
        split = conform_split(split)
        body: Dict[str, Any] = dict(
            include_cols=include_cols,
            col_mapping=col_mapping,
            file_type=ext,
            hf_format=hf_format,
            tagging_schema=tagging_schema,
        )
        body["filter_params"] = {}
        if slice_name:
            slice_ = self.get_slice_by_name(project_name, slice_name)
            body["filter_params"].update(slice_["logic"])

        if filter_params:
            body["filter_params"].update(filter_params)

        if self.get_task_type(project, run) == TaskType.text_multi_label:
            body["task"] = self.get_tasks_for_run(project_name, run_name)[0]

        params = {"inference_name": inference_name}
        url = (
            f"{config.api_url}/{Route.content_path(project, run, split)}/{Route.export}"
        )
        self._export_dataframe_request(url, body, params, file_name)

    def get_project_run_name(
        self, project_id: Optional[UUID4] = None, run_id: Optional[UUID4] = None
    ) -> Tuple[str, str]:
        """Gets the project/run name given project/run IDs, or based on the config's

        Current project and run IDs
        """
        if (project_id and not run_id) or (run_id and not project_id):
            raise GalileoException(
                "You must either provide both the project and run IDs or neither "
                "(using the currently active project/run)"
            )
        if project_id and run_id:
            pid, rid = project_id, run_id
        elif config.current_project_id and config.current_run_id:
            pid, rid = config.current_project_id, config.current_run_id
        else:
            raise GalileoException(
                "You must either provide a project and run name or call "
                "dataquality.init() to initialize a run"
            )
        pname = self.get_project(pid)["name"]
        rname = self.get_project_run(pid, rid)["name"]
        return pname, rname

    def get_run_status(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        pid, rid = self._get_project_run_id(
            project_name=project_name, run_name=run_name
        )
        job_url = f"{config.api_url}/{Route.content_path(pid, rid)}/{Route.latest_job}"
        job = self.make_request(RequestType.GET, job_url)
        return job or {}

    def get_run_link(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> str:
        pid, rid = self._get_project_run_id(
            project_name=project_name, run_name=run_name
        )
        link_url = f"{config.api_url}/{Route.content_path(pid, rid)}/{Route.link}"
        link_data = self.make_request(RequestType.GET, link_url)
        return link_data["link"]

    def wait_for_run(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> None:
        print("Waiting for job (you can safely close this window)...")
        last_progress_message = ""
        while True:
            job = self.get_run_status(project_name=project_name, run_name=run_name)
            if job.get("status") == "completed":
                print(f"Done! Job finished with status {job.get('status')}")
                return
            elif job.get("status") == "failed":
                # Try to properly format the stacktrace
                try:
                    err = json.loads(job.get("error_message", ""))
                except JSONDecodeError:
                    err = job.get("error_message")
                raise GalileoException(
                    f"It seems your run failed with error\n{err}"
                ) from None
            elif job.get("status") == "in_progress" and job.get("progress_message"):
                if last_progress_message != job["progress_message"]:
                    print(f"\t{job['progress_message']}")
                    last_progress_message = job["progress_message"]
                sleep(2)
            elif not job or job.get("status") in ["unstarted", "in_progress"]:
                sleep(2)
            else:
                raise GalileoException(
                    f"It seems there was an issue with your job. Received "
                    f"an unexpected status {job.get('status')}"
                )

    def get_presigned_url(
        self,
        method: str,
        bucket_name: str,
        object_name: str,
        project_id: str,
    ) -> str:
        response = self.make_request(
            request=RequestType.GET,
            url=f"{config.api_url}/{Route.presigned_url}",
            params={
                "api_url": config.api_url,
                "bucket_name": bucket_name,
                "object_name": object_name,
                "method": method.upper(),
                "project_id": project_id,
            },
        )
        return response["url"]

    def get_run_summary(
        self,
        project_name: str,
        run_name: str,
        split: str,
        task: Optional[str] = None,
        inference_name: Optional[str] = None,
        filter_params: Optional[Dict] = None,
    ) -> Dict:
        """Gets overall run summary, or summary of a filtered subset.

        Use filter_params to apply arbitrary filters on the dataframe, based on the
        filter schema:
        https://api.dev.rungalileo.io/redoc#tag/insights
        """
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)
        path = Route.content_path(project, run, split)
        url = f"{config.api_url}/{path}/{Route.summary}"
        params = {}
        if inference_name:
            params["inference_name"] = inference_name
        body = {
            "hard_easy_threshold": True,
            "task": task,
            "filter_params": filter_params or {},
        }
        return self.make_request(RequestType.POST, url, body=body, params=params)

    def get_run_metrics(
        self,
        project_name: str,
        run_name: str,
        split: str,
        task: Optional[str] = None,
        inference_name: Optional[str] = None,
        category: str = "gold",
        filter_params: Optional[Dict] = None,
    ) -> Dict[str, List]:
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)

        all_meta = self.get_metadata_columns(project_name, run_name, split)
        categorical_meta = [i["name"] for i in all_meta["meta"] if i["is_categorical"]]
        avl_cols = categorical_meta + ["gold", "pred"]
        if category not in avl_cols:
            raise GalileoException(
                f"Category must be one of {avl_cols} for this run but got {category}"
            )

        path = Route.content_path(project, run, split)
        url = f"{config.api_url}/{path}/{Route.groupby}"
        params = {"groupby_col": category}
        if inference_name:
            params["inference_name"] = inference_name
        body = {"task": task, "filter_params": filter_params or {}}
        return self.make_request(RequestType.POST, url, body=body, params=params)

    def get_column_distribution(
        self,
        project_name: str,
        run_name: str,
        split: str,
        task: Optional[str] = None,
        inference_name: Optional[str] = None,
        column: str = "data_error_potential",
        filter_params: Optional[Dict] = None,
    ) -> Dict[str, List]:
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)

        all_meta = self.get_metadata_columns(project_name, run_name, split)
        continuous_meta = [i["name"] for i in all_meta["meta"] if i["is_continuous"]]
        avl_cols = continuous_meta + ["data_error_potential"]
        if column not in avl_cols:
            raise GalileoException(
                f"Column must be one of continuous columns {avl_cols} for this run "
                f"but got {column}"
            )

        path = Route.content_path(project, run, split)
        url = f"{config.api_url}/{path}/{Route.distribution}"
        params = {"col": column}
        if inference_name:
            params["inference_name"] = inference_name
        body = {"task": task, "filter_params": filter_params or {}}
        return self.make_request(RequestType.POST, url, body=body, params=params)

    def get_alerts(
        self,
        project_name: str,
        run_name: str,
        split: str,
        inference_name: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Queries API for alerts for a run/split"""
        project, run = self._get_project_run_id(project_name, run_name)
        path = Route.content_path(project, run, split)
        url = f"{config.api_url}/{path}/{Route.alerts}"
        params = {"inference_name": inference_name} if inference_name else None
        return self.make_request(RequestType.GET, url, params=params)

    def delete_alerts_for_split(
        self, project_id: UUID4, run_id: UUID4, split: str
    ) -> None:
        path = Route.content_path(project_id, run_id, split)
        url = f"{config.api_url}/{path}/{Route.alerts}"
        alerts = []
        if split == "inference":
            inference_names = self.get_inference_names(project_id, run_id)
            for inf_name in inference_names["inference_names"]:
                params = {"inference_name": inf_name}
                res = self.make_request(RequestType.GET, url, params=params)
                alerts.extend([alert["id"] for alert in res])
        else:
            res = self.make_request(RequestType.GET, url)
            alerts.extend([alert["id"] for alert in res])
        path = Route.content_path(project_id, run_id, split)
        for alert_id in alerts:
            url = f"{config.api_url}/{path}/{Route.alerts}/{alert_id}"
            self.make_request(RequestType.DELETE, url)

    def delete_alerts(
        self,
        project_name: str,
        run_name: str,
    ) -> None:
        """Delete all alerts for a run"""
        project_id, run_id = self._get_project_run_id(project_name, run_name)
        for split in self.get_splits(project_id, run_id)["splits"]:
            self.delete_alerts_for_split(project_id, run_id, split)
        print(f"All alerts removed for run {project_name}/{run_name}")

    def get_edits(
        self,
        project_name: str,
        run_name: str,
        split: str,
        inference_name: Optional[str] = None,
    ) -> List:
        """Gets all edits for a run/split"""
        project, run = self._get_project_run_id(project_name, run_name)
        split = conform_split(split)

        url = (
            f"{config.api_url}/{Route.content_path(project, run, split)}/{Route.edits}"
        )
        params = {"inference_name": inference_name} if inference_name else None
        return self.make_request(RequestType.GET, url, params=params)

    def export_edits(
        self,
        project_name: str,
        run_name: str,
        split: str,
        file_name: str,
        inference_name: Optional[str] = None,
        include_cols: Optional[List[str]] = None,
        col_mapping: Optional[Dict[str, str]] = None,
        hf_format: bool = False,
        tagging_schema: Optional[TaggingSchema] = None,
    ) -> None:
        """Export the edits of a project/run/split to disk as a file

        :param project_name: The project name
        :param run_name: The run name
        :param split: The split to export on
        :param file_name: The file name. Must end in a supported FileType
        :param inference_name: Required if split is inference. The name of the inference
            split to get data for.
        :param include_cols: List of columns to include in the export. If not set,
        all columns will be exported.
        :param col_mapping: Dictionary of renamed column names for export.
        :param hf_format: (NER only)
            Whether to export the dataframe in a HuggingFace compatible format
        :param tagging_schema: (NER only)
            If hf_format is True, you must pass a tagging schema
        :param filter_params: Filters to apply to the dataframe before exporting. Only
        rows with matching filters will be included in the exported data. If a slice
        """
        edits = self.get_edits(project_name, run_name, split, inference_name)

        ext = os.path.splitext(file_name)[-1].lstrip(".")
        assert ext in list(FileType), f"File must be one of {list(FileType)}"

        body: Dict[str, Any] = dict(
            include_cols=include_cols,
            col_mapping=col_mapping,
            file_type=ext,
            hf_format=hf_format,
            tagging_schema=tagging_schema,
            edit_ids=[edit["id"] for edit in edits],
        )
        url = f"{config.api_url}/{Route.export_edits}"
        params = {"inference_name": inference_name}
        self._export_dataframe_request(url, body, params, file_name)

    def _export_dataframe_request(
        self, url: str, body: Dict, params: Dict, file_name: str
    ) -> None:
        token = self.get_token()
        with requests.post(
            url, json=body, stream=True, headers=headers(token), params=params
        ) as r:
            self._validate_response(r)
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    def notify_email(
        self, data: Dict, template: str, emails: Optional[List[str]] = None
    ) -> None:
        self.make_request(
            RequestType.POST,
            url=f"{config.api_url}/{Route.notify}",
            body={"data": data, "template": template, "emails": emails},
        )

    def get_splits(self, project_id: UUID4, run_id: UUID4) -> Dict:
        return self.make_request(
            RequestType.GET,
            url=(
                f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}/"
                f"{Route.splits}"
            ),
        )

    def get_inference_names(self, project_id: UUID4, run_id: UUID4) -> Dict:
        return self.make_request(
            RequestType.GET,
            url=(
                f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}/"
                f"{Route.inference_names}"
            ),
        )

    def set_metric_for_run(self, project_id: UUID4, run_id: UUID4, data: Dict) -> Dict:
        return self.make_request(
            RequestType.PUT,
            url=(
                f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}/"
                f"{Route.metrics}"
            ),
            body=data,
        )

    def get_healthcheck_dq(self) -> Dict:
        return self.make_request(
            RequestType.GET, url=f"{config.api_url}/{Route.healthcheck_dq}"
        )

    def upload_file_for_project(
        self,
        project_id: str,
        file_path: str,
        export_format: str,
        export_cols: List[str],
        bucket: str,
    ) -> Any:
        url = f"{config.api_url}/{Route.projects}/{project_id}/{Route.upload_file}"
        res = self.make_request(
            return_response_without_validation=True,
            request=RequestType.POST,
            url=url,
            files={
                "file": (
                    os.path.basename(file_path),
                    open(file_path, "rb"),
                    "application/octet-stream",
                ),
                "upload_metadata": (
                    None,
                    json.dumps(
                        {
                            "export_format": export_format,
                            "export_cols": export_cols,
                            "bucket": bucket,
                        }
                    ),
                    "application/json",
                ),
            },
        )
        return res

    def get_presigned_url_for_model(
        self, project_id: UUID4, run_id: UUID4, model_kind: str, model_parameters: Dict
    ) -> str:
        """
        Returns a presigned url for uploading a model to S3

        """
        return self.make_request(
            RequestType.POST,
            url=f"{config.api_url}/{Route.projects}/{str(project_id)}/{Route.runs}/{str(run_id)}/{Route.model}",
            body={"kind": model_kind, "parameters": model_parameters},
        )["upload_url"]

    def get_uploaded_model_info(self, project_id: UUID4, run_id: UUID4) -> Any:
        """
        Returns information about the model for a given run.
        Will also update the status to complete.
        :param project_id: The project id
        :param run_id: The run id
        """
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{str(project_id)}/{Route.runs}/{str(run_id)}/{Route.model}",
        )
