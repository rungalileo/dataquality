from typing import Dict

from pydantic.types import UUID4

from dataquality.core.config import config
from dataquality.utils.name import random_name


class _Init:
    def create_new_project_and_run(self) -> Dict:
        pass

    def get_project(self) -> Dict:
        pass

    def get_project_and_start_new_run(self, project_name: str) -> Dict:
        pass

    def get_run_from_project(self, project_name: str, run_id: UUID4) -> Dict:
        pass


def init(project_name: str, run_id: str) -> None:
    _init = _Init()
    if project_name is None and run_id is None:
        # no project and no run id, start a new project and start a new run
        print(f"✨ Initializing project {project_name}")
    elif project_name is not None and run_id is None:
        # if project exists, start new run
        print(f"📡 Retrieving project, {project_name}, and starting a new run")
        # otherwise create project with given name and start new run
        print(f"✨ Project {project_name} was not found. Initializing a Galileo project")
    elif project_name is not None and run_id is not None:
        # given a project and run retrieve the data and set info to state
        print(f"📡 Retrieving existing run from project, {project_name}")
    else:
        print("⚠️ Please specify a project name to initialize a new Galileo run.")
    config.update_file_config()
