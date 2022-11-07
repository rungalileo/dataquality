import os
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

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
    def __check_login(self) -> None:
        if not config.token:
            raise GalileoException("You are not logged in. Call dataquality.login()")

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
        self.__check_login()
        return self.get_current_user()["id"]

    def get_current_user(self) -> Dict:
        if not config.token:
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

    def make_request(
        self,
        request: RequestType,
        url: str,
        body: Dict = None,
        data: Dict = None,
        params: Dict = None,
        header: Dict = None,
        timeout: Union[int, None] = None,
    ) -> Any:
        """Makes an HTTP request.

        This is the center point of all functions and the main entry/exit for the
        dataquality client to interact with the server.
        """
        self.__check_login()
        header = header or headers(config.token)
        res = RequestType.get_method(request.value)(
            url, json=body, params=params, headers=header, data=data, timeout=timeout
        )
        self._validate_response(res)
        return res.json()

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

    def create_project(self, project_name: str, is_public: bool = True) -> Dict:
        """Creates a project given a name and returns the project information"""
        body = {"name": project_name, "is_public": is_public}
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
        self, project_name: str = None, run_name: str = None, task: str = None
    ) -> List[str]:
        """Gets the labels for a given run, else the currently initialized project/run

        If you do not provide a project and run name, the currently initialized
        project/run will be used. Otherwise you must provide both a project and run name
        If the run is a multi-label run, a task must be provided
        """
        project, run = self._get_project_run_id(
            project_name=project_name, run_name=run_name
        )
        task_type = self.get_task_type(project, run)
        if not task and task_type == TaskType.text_multi_label:
            raise GalileoException("For multi-label runs, a task name must be provided")

        url = f"{config.api_url}/{Route.content_path(project, run)}/{Route.labels}"
        params = {"task": task} if task else None
        res = self.make_request(RequestType.GET, url=url, params=params)
        return res["labels"]

    def get_tasks_for_run(
        self, project_name: str = None, run_name: str = None
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
        project_name: str = None,
        run_name: str = None,
        labels: Union[List, List[List]] = None,
    ) -> Dict:
        """Reinitiate a project/run that has already been finished

        If a project and run name have been provided, that project/run will be
        reinitiated, otherwise we trigger the currently initialized project/run.

        This will clear out the current state in the server, and will recalculate
        * DEP score
        * UMAP Embeddings for visualization
        * Smart features

        :param project_name: If not set, will use the currently active project
        :param run_name: If not set, will use the currently active run
        :param labels: If set, will reprocess the run with these labels. If not set,
        labels will be used from the previously processed run. These must match the
        labels that were originally logged
        """
        project, run = self._get_project_run_id(project_name, run_name)
        project_name = project_name or self.get_project(project)["name"]
        run_name = run_name or self.get_project_run(project, run)["name"]
        task_type = self.get_task_type(project, run)

        # Multi-label has tasks and List[List] for labels
        if task_type == TaskType.text_multi_label:
            tasks = self.get_tasks_for_run(project_name, run_name)
            if not labels:
                labels = [
                    self.get_labels_for_run(project_name, run_name, t) for t in tasks
                ]
        else:
            tasks = []
            if not labels:
                try:
                    labels = self.get_labels_for_run(project_name, run_name)
                except GalileoException as e:
                    if "No data found" in str(e):
                        e = GalileoException(
                            f"It seems no data is available for run "
                            f"{project_name}/{run_name}"
                        )
                    raise e from None
                # There were no labels available for this run
                except KeyError:
                    raise GalileoException(
                        "It seems we cannot find the labels for this run. Please call "
                        "api_client.reprocess_run again, passing in your labels to the "
                        "'labels' keyword"
                    ) from None

        body = dict(
            project_id=str(project),
            run_id=str(run),
            labels=labels,
            tasks=tasks or None,
            task_type=task_type,
            xray=False,  # Don't recalculate XRay in process
        )
        res = self.make_request(
            RequestType.POST, url=f"{config.api_url}/{Route.jobs}", body=body
        )
        print(
            f"Job {res['job_name']} successfully resubmitted. New results will be "
            f"available soon at {res['link']}"
        )
        return res

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
        return self.make_request(RequestType.GET, url)

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
        filter_params: Dict = None,
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
        all columns will be exported.
        :param col_mapping: Dictionary of renamed column names for export.
        :param hf_format: (NER only)
            Whether to export the dataframe in a HuggingFace compatible format
        :param tagging_schema: (NER only)
            If hf_format is True, you must pass a tagging schema
        :param filter_params: Filters to apply to the dataframe before exporting. Only
        rows with matching filters will be included in the exported data. If a slice

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
        list_jobs_url = f"{config.api_url}/{Route.content_path(pid, rid)}/{Route.jobs}"
        jobs = self.make_request(RequestType.GET, list_jobs_url)
        sorted_jobs = sorted(jobs, key=lambda row: row["created_at"], reverse=True)
        return sorted_jobs[0] if sorted_jobs else {}

    def wait_for_run(
        self, project_name: Optional[str] = None, run_name: Optional[str] = None
    ) -> None:
        print("Waiting for job...")
        while True:
            job = self.get_run_status(project_name=project_name, run_name=run_name)
            if job.get("status") == "completed":
                print(f"Done! Job finished with status {job.get('status')}")
                return
            elif job.get("status") == "failed":
                raise GalileoException(
                    f"It seems your run failed with status "
                    f"{job.get('status')}, error {job.get('error_message')}"
                )
            elif not job or job.get("status") in ["unstarted", "in_progress"]:
                sleep(2)
            else:
                raise GalileoException(
                    f"It seems there was an issue with your job. Received "
                    f"an unexpected status {job.get('status')}"
                )

    def get_presigned_url(
        self, project_id: str, method: str, bucket_name: str, object_name: str
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
        task: str = None,
        inference_name: str = None,
        filter_params: Dict = None,
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
        task: str = None,
        inference_name: str = None,
        category: str = "gold",
        filter_params: Dict = None,
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
        task: str = None,
        inference_name: str = None,
        column: str = "data_error_potential",
        filter_params: Dict = None,
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

    def get_xray_cards(
        self, project_name: str, run_name: str, split: str, inference_name: str = None
    ) -> List[Dict[str, str]]:
        """Queries API for xray cards for a run/split"""
        project, run = self._get_project_run_id(project_name, run_name)
        path = Route.content_path(project, run, split)
        url = f"{config.api_url}/{path}/{Route.xray}"
        params = {"inference_name": inference_name} if inference_name else None
        return self.make_request(RequestType.GET, url, params=params)

    def get_edits(
        self, project_name: str, run_name: str, split: str, inference_name: str = None
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
        inference_name: str = None,
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

    def send_analytics(
        self,
        project_id: str = "UNKNOWN",
        run_id: str = "UNKNOWN",
        run_task_type: str = "UNKNOWN",
        payload: Dict = {},
        scope: str = "dq",
    ) -> Dict:
        """Creates a project given a name and returns the project information"""
        path = Route.content_path(project_id, run_id)
        body = {**payload, "run_task_type": run_task_type}
        url = f"{config.api_url}/{path}/{Route.ampli}/{scope}"

        return self.make_request(RequestType.POST, url, body=body, timeout=1)

    def _export_dataframe_request(
        self, url: str, body: Dict, params: Dict, file_name: str
    ) -> None:
        with requests.post(
            url, json=body, stream=True, headers=headers(config.token), params=params
        ) as r:
            self._validate_response(r)
            with open(file_name, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
