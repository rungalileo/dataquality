import os
import shutil
import warnings
from typing import Dict, Optional

from pydantic.types import UUID4

import dataquality
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.core.auth import login
from dataquality.exceptions import GalileoException
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.name import random_name

api_client = ApiClient()


class _Init:
    def get_project_by_name_for_user(self, project_name: str) -> Dict:
        return api_client.get_project_by_name(project_name)

    def get_project_run_by_name_for_user(
        self, project_name: str, run_name: str
    ) -> Dict:
        return api_client.get_project_run_by_name(project_name, run_name)

    def _initialize_new_project(
        self, project_name: str, is_public: bool = True
    ) -> Dict:
        visability = "public" if is_public else "private"
        print(f"✨ Initializing {visability} project {project_name}")
        try:
            return api_client.create_project(
                project_name=project_name, is_public=is_public
            )
        except GalileoException as e:
            if "A project with this name already exists" in str(e):
                return api_client.get_project_by_name(project_name)
            else:
                raise e

    def _initialize_run_for_project(
        self, project_name: str, run_name: str, task_type: TaskType
    ) -> Dict:
        print(f"🏃‍♂️ Starting run {run_name}")
        return api_client.create_run(project_name, run_name, task_type)

    def create_log_file_dir(
        self, project_id: UUID4, run_id: UUID4, overwrite_local: bool
    ) -> None:
        write_output_dir = f"{BaseGalileoLogger.LOG_FILE_DIR}/{project_id}/{run_id}"
        if overwrite_local and os.path.exists(write_output_dir):
            shutil.rmtree(write_output_dir)
        if not os.path.exists(write_output_dir):
            os.makedirs(write_output_dir)


def init(
    task_type: str,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    is_public: bool = True,
    overwrite_local: bool = True,
) -> None:
    """
    Start a run

    Initialize a new run and new project, initialize a new run in an existing project,
    or reinitialize an existing run in an existing project.

    Optionally provide project and run names to create a new project/run or restart
    existing ones.

    :param task_type: The task type for modeling. This must be one of the valid
    `dataquality.schemas.task_type.TaskType` options
    :param project_name: The project name. If not passed in, a random one will be
    generated. If provided, and the project does not exist, it will be created. If it
    does exist, it will be set.
    :param run_name: The run name. If not passed in, a random one will be
    generated. If provided, and the project does not exist, it will be created. If it
    does exist, it will be set.
    :param is_public: Boolean value that sets the project's visibility. Default True.
    :param overwrite_local: If True, the current project/run log directory will be
    cleared during this function. If logging over many sessions with checkpoints, you
    may want to set this to False. Default True
    """
    if not api_client.valid_current_user():
        login()
    _init = _Init()
    BaseGalileoLogger.validate_task(task_type)
    task_type = TaskType[task_type]
    config.task_type = task_type
    if not project_name and not run_name:
        # no project and no run id, start a new project and start a new run
        project_name, run_name = random_name(), random_name()
        project_response = _init._initialize_new_project(
            project_name=project_name, is_public=is_public
        )
        run_response = _init._initialize_run_for_project(
            project_name=project_name, run_name=run_name, task_type=task_type
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
                project_name=project_name, run_name=run_name, task_type=task_type
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
            project_response = _init._initialize_new_project(
                project_name=project_name, is_public=is_public
            )
            run_response = _init._initialize_run_for_project(
                project_name=project_name, run_name=run_name, task_type=task_type
            )
            config.current_project_id = project_response["id"]
            config.current_run_id = run_response["id"]
    elif project_name and run_name:
        project = _init.get_project_by_name_for_user(project_name)
        # if project actually exists, get the run
        if project.get("name"):
            # If the project and run exist, connect to them
            print(f"📡 Retrieving run from existing project, {project_name}")
            run = _init.get_project_run_by_name_for_user(
                project["name"], run_name=run_name
            )
            if run.get("id"):
                config.current_project_id = project["id"]
                config.current_run_id = run["id"]
                warnings.warn(
                    f"Run: {project_name}/{run_name} already exists! "
                    "The existing run will get overwritten on call to finish()!"
                )
                print(f"🛰 Connected to project, {project_name}, and run, {run_name}.")
            else:
                # If the run does not exist, create it
                run_response = _init._initialize_run_for_project(
                    project_name, run_name, task_type
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
            project_response = _init._initialize_new_project(
                project_name=project_name, is_public=is_public
            )
            run_response = _init._initialize_run_for_project(
                project_name=project_name, run_name=run_name, task_type=task_type
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
        _init.create_log_file_dir(
            config.current_project_id,
            config.current_run_id,
            overwrite_local=overwrite_local,
        )
    # Reset all config variables
    dataquality.get_data_logger().logger_config.reset()
