import os
import warnings
from typing import Dict, List, Optional

import requests
from pydantic.types import UUID4

from dataquality import config
from dataquality.core.auth import _Auth
from dataquality.core.log import JsonlLogger
from dataquality.exceptions import GalileoException
from dataquality.schemas import Route
from dataquality.utils.auth import headers
from dataquality.utils.name import random_name


class _Init:
    def create_project(self, data: Dict) -> Dict:
        if not config.token:
            raise GalileoException("Token not present, please log in!")
        req = requests.post(
            f"{config.api_url}/{Route.projects}",
            json=data,
            headers=headers(config.token),
        )
        return req.json()

    def create_project_run(self, project_id: UUID4, data: Dict) -> Dict:
        if not config.token:
            raise GalileoException("Token not present, please log in!")
        req = requests.post(
            f"{config.api_url}/{Route.projects}/{project_id}/runs",
            json=data,
            headers=headers(config.token),
        )
        return req.json()

    def get_user_projects(self) -> List[Dict]:
        user_id = self.get_user_id()
        req = requests.get(
            f"{config.api_url}/{Route.users}/{user_id}/projects",
            headers=headers(config.token),
        )
        return req.json()

    def get_project_by_name_for_user(self, project_name: str) -> Dict:
        projects = self.get_user_projects()
        name_project = {project["name"]: project for project in projects}
        return name_project.get(project_name, {})

    def get_runs_from_project_for_user(self, project_name: str) -> Dict:
        """Gets the runs for a given project for a given user"""
        project = self.get_project_by_name_for_user(project_name)
        if not project:
            return {}
        pid = project.get("id")
        req = requests.get(
            f"{config.api_url}/{Route.projects}/{pid}/runs",
            headers=headers(config.token),
        )
        return req.json()

    def get_project_run_by_name_for_user(
        self, project_name: str, run_name: str
    ) -> Dict:
        runs = self.get_runs_from_project_for_user(project_name)
        name_run = {run["name"]: run for run in runs}
        return name_run.get(run_name, {})

    def get_run_from_project(self, project_id: UUID4, run_id: UUID4) -> Dict:
        if not config.token:
            raise GalileoException("Token not present, please log in!")
        return requests.get(
            f"{config.api_url}/{Route.projects}/{project_id}/runs/{run_id}",
            headers=headers(config.token),
        ).json()

    def get_user_id(self) -> UUID4:
        if not config.token:
            raise GalileoException("Token not present, please log in!")
        _auth = _Auth(config=config, auth_method=config.auth_method)
        return _auth.get_current_user(config)["id"]

    def _initialize_new_project(self, project_name: str) -> Dict:
        print(f"✨ Initializing project {project_name}")
        body = {"name": project_name}
        return self.create_project(body)

    def _initialize_run_for_project(self, project_id: UUID4, run_name: str) -> Dict:
        print(f"🏃‍♂️ Starting run {run_name}")
        body = {"name": run_name}
        return self.create_project_run(project_id, body)

    def create_log_file_dir(self, project_id: UUID4, run_id: UUID4) -> None:
        write_output_dir = f"{JsonlLogger.LOG_FILE_DIR}/{project_id}/{run_id}"
        if not os.path.exists(write_output_dir):
            os.makedirs(write_output_dir)


def init(project_name: Optional[str] = None, run_name: Optional[str] = None) -> None:
    """
    Start a run

    Initialize a new run and new project, initialize a new run in an existing project,
    or reinitialize an existing run in an existing project.

    Optionally provide project and run names to create a new project/run or restart
    existing ones.

    :param project_name: The project name. If not passed in, a random one will be
    generated. If provided, and the project does not exist, it will be created. If it
    does exist, it will be set.
    :param run_name: The run name. If not passed in, a random one will be
    generated. If provided, and the project does not exist, it will be created. If it
    does exist, it will be set.
    """
    _init = _Init()
    config.labels = None
    if not project_name and not run_name:
        # no project and no run id, start a new project and start a new run
        project_name, run_name = random_name(), random_name()
        project_response = _init._initialize_new_project(project_name=project_name)
        run_response = _init._initialize_run_for_project(
            project_id=project_response["id"], run_name=run_name
        )
        config.current_project_id = project_response["id"]
        config.current_run_id = run_response["id"]
        print(f"🛰 Created project, {project_name}, and new run, {run_name}.")
    elif project_name and not run_name:
        project = _init.get_project_by_name_for_user(project_name)
        # if project exists, start new run
        if project.get("id") is not None:
            run_name = random_name()
            print(f"📡 Retrieved project, {project_name}, and starting a new run")
            run_response = _init._initialize_run_for_project(
                project_id=project["id"], run_name=run_name
            )
            config.current_project_id = project["id"]
            config.current_run_id = run_response["id"]
            print(
                f"🛰 Connected to project, {project_name}, and created run, {run_name}."
            )
        # otherwise create project with given name and start new run
        else:
            print(f"💭 Project {project_name} was not found.")
            run_name = random_name()
            project_response = _init._initialize_new_project(project_name=project_name)
            run_response = _init._initialize_run_for_project(
                project_id=project_response["id"], run_name=run_name
            )
            config.current_project_id = project_response["id"]
            config.current_run_id = run_response["id"]
    elif project_name and run_name:
        # If the project and run exist, connect to them
        print(f"📡 Retrieving existing run from project, {project_name}")
        project = _init.get_project_by_name_for_user(project_name)
        # if project actually exists, get the run
        if project.get("name"):
            run = _init.get_project_run_by_name_for_user(
                project["name"], run_name=run_name
            )
            if run.get("id"):
                config.current_project_id = project["id"]
                config.current_run_id = run["id"]
                print(f"🛰 Connected to project, {project_name}, and run, {run_name}.")
            else:
                # If the run does not exist, create it
                run_response = _init._initialize_run_for_project(
                    project["id"], run_name
                )
                config.current_project_id = project["id"]
                config.current_run_id = run_response["id"]
                print(
                    f"🛰 Connected to project, {project['name']} "
                    f"and created new run, {run_name}."
                )
        else:
            # User gave us a new project name and new run name to create, so create it
            print(f"💭 Project {project_name} was not found.")
            project_response = _init._initialize_new_project(project_name=project_name)
            run_response = _init._initialize_run_for_project(
                project_id=project_response["id"], run_name=run_name
            )
            config.current_project_id = project_response["id"]
            config.current_run_id = run_response["id"]
            print(f"🛰 Created project, {project_name}, and new run, {run_name}.")
    else:
        # The user provided a run name and no project name. No good
        warnings.warn(
            "⚠️ You must specify a project name to initialize or create a new Galileo "
            "run. Add a project name, or simply run dataquality.init()."
        )
        return
    config.update_file_config()
    if config.current_project_id and config.current_run_id:
        _init.create_log_file_dir(config.current_project_id, config.current_run_id)
