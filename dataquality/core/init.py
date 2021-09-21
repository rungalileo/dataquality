import os
from typing import Dict, List, Optional

import requests
from pydantic.types import UUID4

from dataquality.core.auth import _Auth
from dataquality.core.config import Config, config
from dataquality.core.log import JsonlLogger
from dataquality.schemas import Route
from dataquality.utils.auth import headers
from dataquality.utils.name import random_name


class _Init:
    def create_project(self, data: Dict, config: Config) -> Dict:
        if not config.token:
            raise Exception("Token not present, please log in!")
        req = requests.post(
            f"{config.api_url}/{Route.projects}",
            json=data,
            headers=headers(config.token),
        )
        return req.json()

    def create_project_run(self, project_id: UUID4, data: Dict, config: Config) -> Dict:
        if not config.token:
            raise Exception("Token not present, please log in!")
        req = requests.post(
            f"{config.api_url}/{Route.projects}/{project_id}/runs",
            json=data,
            headers=headers(config.token),
        )
        return req.json()

    def get_user_projects(self, user_id: UUID4) -> List[Dict]:
        if not config.token:
            raise Exception("Token not present, please log in!")
        req = requests.get(
            f"{config.api_url}/{Route.users}/{user_id}/projects",
            headers=headers(config.token),
        )
        return req.json()

    def get_project_by_name_for_user(self, user_id: UUID4, project_name: str) -> Dict:
        projects = self.get_user_projects(user_id)
        return [project for project in projects if project["name"] == project_name][0]

    def get_run_from_project(
        self, config: Config, project_id: UUID4, run_id: UUID4
    ) -> Dict:
        if not config.token:
            raise Exception("Token not present, please log in!")
        return requests.get(
            f"{config.api_url}/{Route.projects}/{project_id}/runs/{run_id}",
            headers=headers(config.token),
        ).json()

    def get_user_id(self, _auth: _Auth, config: Config) -> UUID4:
        return _auth.get_current_user(config)["id"]

    def _initialize_new_project(self, config: Config, project_name: str) -> Dict:
        print(f"✨ Initializing project {project_name}")
        body = {"name": project_name}
        return self.create_project(body, config)

    def _initialize_run_for_project(
        self, config: Config, project_id: UUID4, run_name: str
    ) -> Dict:
        print(f"🏃‍♂️ Starting run {run_name}")
        body = {"name": run_name}
        return self.create_project_run(project_id, body, config)

    def create_log_file_dir(self, project_id: UUID4, run_id: UUID4) -> None:
        write_output_dir = f"{JsonlLogger.LOG_FILE_DIR}/{project_id}/{run_id}"
        if not os.path.exists(write_output_dir):
            os.makedirs(write_output_dir)


def init(project_name: Optional[str] = None, run_id: Optional[UUID4] = None) -> None:
    _auth = _Auth(config=config, auth_method=config.auth_method)
    _init = _Init()
    config.labels = None
    if project_name is None and run_id is None:
        # no project and no run id, start a new project and start a new run
        project_name, run_name = random_name(), random_name()
        project_response = _init._initialize_new_project(
            config=config, project_name=project_name
        )
        run_response = _init._initialize_run_for_project(
            config=config, project_id=project_response["id"], run_name=run_name
        )
        config.current_project_id = project_response["id"]
        config.current_run_id = run_response["id"]
        print(f"🛰 Created project, {project_name}, and new run, {run_name}.")
    elif project_name is not None and run_id is None:
        user_id = _init.get_user_id(_auth, config)
        project = _init.get_project_by_name_for_user(user_id, project_name)
        # if project exists, start new run
        if project.get("id") is not None:
            run_name = random_name()
            print(f"📡 Retrieved project, {project_name}, and starting a new run")
            run_response = _init._initialize_run_for_project(
                config=config, project_id=project["id"], run_name=run_name
            )
            config.current_project_id = project["id"]
            config.current_run_id = run_response["id"]
            print(
                f"🛰 Connected to project, {project_name}, and created run, {run_name}."
            )
        # otherwise create project with given name and start new run
        else:
            print(f"💭 Project {project_name} was not found.")
            project_name, run_name = random_name(), random_name()
            project_response = _init._initialize_new_project(
                config=config, project_name=project_name
            )
            run_response = _init._initialize_run_for_project(
                config=config, project_id=project_response["id"], run_name=run_name
            )
            config.current_project_id = project_response["id"]
            config.current_run_id = run_response["id"]
    elif project_name is not None and run_id is not None:
        # given a project and run, retrieve the data and set info to state
        print(f"📡 Retrieving existing run from project, {project_name}")
        user_id = _init.get_user_id(_auth, config)
        project = _init.get_project_by_name_for_user(user_id, project_name)
        # if project actually exists, get the run
        if project.get("id") is not None:
            run = _init.get_run_from_project(
                config=config, project_id=project["id"], run_id=run_id
            )
            if run.get("id") is not None:
                config.current_project_id = project["id"]
                config.current_run_id = run["id"]
                print(
                    f"🛰 Connected to project, {project_name}, and run, {run['name']}."
                )
            else:
                print(
                    f"⚠️ The Galileo account associated with {config.current_user}"
                    f" does not have a run with id {run_id} "
                    f"associated with the project {project_name}."
                )
                return
        else:
            print(
                f"⚠️ The Galileo account associated with {config.current_user}"
                f" does not have a project named {project_name}."
            )
            return
    else:
        print(
            "⚠️ You must specify a project name to initialize a new Galileo run"
            " or simply run dataquality.init()."
        )
        return
    config.update_file_config()
    if config.current_project_id and config.current_run_id:
        _init.create_log_file_dir(config.current_project_id, config.current_run_id)
