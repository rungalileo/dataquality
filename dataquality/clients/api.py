from typing import Any, Dict, List

from pydantic.types import UUID4

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.schemas import RequestType, Route
from dataquality.utils.auth import headers


class ApiClient:
    def __check_login(self) -> None:
        if not config.token:
            raise GalileoException("You are not logged in. Call dataquality.login()")

    def _get_user_id(self) -> UUID4:
        self.__check_login()
        return self.get_current_user()["id"]

    def get_current_user(self) -> Dict:
        if not config.token:
            raise GalileoException("Current user is not set!")

        return self.make_request(
            RequestType.GET, url=f"{config.api_url}/{Route.current_user}"
        )

    def make_request(
        self,
        request: RequestType,
        url: str,
        body: Dict = None,
        data: Dict = None,
        params: Dict = None,
        header: Dict = None,
    ) -> Any:
        """Makes an HTTP request.

        This is the center point of all functions and the main entry/exit for the
        dataquality client to interact with the server.
        """
        self.__check_login()
        header = header or headers(config.token)
        req = RequestType.get_method(request.value)(
            url, json=body, params=params, headers=header, data=data
        )
        if not req.ok:
            msg = (
                "Something didn't go quite right. The api returned a non-ok status "
                f"code {req.status_code} with output: {req.text}"
            )
            raise GalileoException(msg)
        return req.json()

    def get_projects(self) -> List[Dict]:
        user_id = self._get_user_id()
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.users}/{user_id}/{Route.projects}",
        )

    def get_project_by_name(self, project_name: str) -> Dict:
        projs = self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}?project_name={project_name}",
        )
        return projs[0] if projs else {}

    def get_project_runs(self, project_id: UUID4) -> List[Dict]:
        """Gets all runs from a project by ID"""
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/",
        )

    def get_project_runs_by_name(self, project_name: str) -> List[Dict]:
        """Gets all runs from a project by name"""
        proj = self.get_project_by_name(project_name)
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}/",
        )

    def get_project_run(self, project_id: UUID4, run_id: UUID4) -> Dict:
        """Gets a run in a project by ID"""
        return self.make_request(
            RequestType.GET,
            url=f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}",
        )

    def get_project_run_by_name(self, project_name: str, run_name: str) -> Dict:
        proj = self.get_project_by_name(project_name)
        url = (
            f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}?"
            f"run_name={run_name}"
        )
        runs = self.make_request(RequestType.GET, url=url)
        return runs[0] if runs else {}

    def create_project(self, project_name: str) -> Dict:
        """Creates a project given a name and returns the project information"""
        body = {"name": project_name}
        return self.make_request(
            RequestType.POST, url=f"{config.api_url}/{Route.projects}", body=body
        )

    def create_run(self, project_name: str, run_name: str) -> Dict:
        """Creates a run in a given project"""
        body = {"name": run_name}
        proj = self.get_project_by_name(project_name)
        return self.make_request(
            RequestType.POST,
            url=f"{config.api_url}/{Route.projects}/{proj['id']}/{Route.runs}",
            body=body,
        )

    def reset_run(self, project_id: UUID4, run_id: UUID4) -> Dict:
        """Resets a run by clearing all minio run data.

        Called before any call to `dataquality.finish`
        """
        url = (
            f"{config.api_url}/{Route.projects}/{project_id}/{Route.runs}/{run_id}/data"
        )
        return self.make_request(RequestType.DELETE, url=url)

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
                f"No project/run found with name " f"{project_name}/{run_name}"
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


api_client = ApiClient()
