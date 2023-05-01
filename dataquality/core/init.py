import os
import shutil
import warnings
from typing import Dict, Optional, Tuple

from pydantic.types import UUID4
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

import dataquality
from dataquality.clients.api import ApiClient
from dataquality.core._config import (
    EXOSCALE_FQDN_SUFFIX,
    GALILEO_DEFAULT_IMG_BUCKET_NAME,
    GALILEO_DEFAULT_RESULT_BUCKET_NAME,
    GALILEO_DEFAULT_RUN_BUCKET_NAME,
    _check_dq_version,
    config,
)
from dataquality.core.auth import login
from dataquality.exceptions import GalileoException, GalileoWarning
from dataquality.loggers import BaseGalileoLogger
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dq_logger import DQ_LOG_FILE_HOME
from dataquality.utils.helpers import check_noop
from dataquality.utils.name import validate_name

api_client = ApiClient()


class InitManager:
    @retry(
        retry=retry_if_exception_type(GalileoException),
        wait=wait_exponential_jitter(initial=0.1, max=2),
        stop=stop_after_attempt(5),
    )
    def get_or_create_project(
        self, project_name: str, is_public: bool
    ) -> Tuple[Dict, bool]:
        """Gets a project by name, or creates a new one if it doesn't exist.

        Returns:
            Tuple[Dict, bool]: The project and a boolean indicating if the project
            was created
        """
        project = api_client.get_project_by_name(project_name)
        created = False
        if not project:
            project = api_client.create_project(project_name, is_public=is_public)
            created = True

        visibility = "public" if is_public else "private"
        created_str = "new" if created else "existing"
        print(f"✨ Initializing {created_str} {visibility} project '{project_name}'")
        return project, created

    def get_or_create_run(
        self, project_name: str, run_name: str, task_type: TaskType
    ) -> Tuple[Dict, bool]:
        """Gets a run by name, or creates a new one if it doesn't exist.

        Returns:
            Tuple[Dict, bool]: The run and a boolean indicating if the run was created
        """
        run = api_client.get_project_run_by_name(project_name, run_name)
        created = False
        if not run:
            run = api_client.create_run(project_name, run_name, task_type=task_type)
            created = True

        created_str = "new" if created else "existing"
        verb = "Creating" if created else "Fetching"
        print(f"🏃‍♂️ {verb} {created_str} run '{run_name}'")
        return run, created

    def create_log_file_dir(
        self, project_id: UUID4, run_id: UUID4, overwrite_local: bool
    ) -> None:
        write_output_dir = f"{BaseGalileoLogger.LOG_FILE_DIR}/{project_id}/{run_id}"
        stdout_dir = f"{DQ_LOG_FILE_HOME}/{run_id}"
        for out_dir in [write_output_dir, stdout_dir]:
            if overwrite_local and os.path.exists(out_dir):
                shutil.rmtree(out_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)


@check_noop
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

    Before creating the project, check:
    - The user is valid, login if not
    - The DQ client version is compatible with API version

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
    _check_dq_version()
    _init = InitManager()
    task_type = BaseGalileoLogger.validate_task(task_type)
    config.task_type = task_type
    if not project_name and run_name:
        # The user provided a run name and no project name. No good
        warnings.warn(
            "⚠️ You must specify a project name to initialize or create a new Galileo "
            "run. Add a project name, or simply run dq.init()."
        )
        return

    project_name = validate_name(project_name, assign_random=True)
    run_name = validate_name(run_name, assign_random=True)

    project, proj_created = _init.get_or_create_project(project_name, is_public)
    run, run_created = _init.get_or_create_run(project_name, run_name, task_type)

    if not run_created:
        warnings.warn(
            f"Run: {project_name}/{run_name} already exists! "
            "The existing run will get overwritten on call to finish()!",
            GalileoWarning,
        )

    config.current_project_id = project["id"]
    config.current_run_id = run["id"]
    _dq_healthcheck_response = api_client.get_healthcheck_dq()
    _bucket_names = _dq_healthcheck_response.get("bucket_names", {})
    config.root_bucket_name = _bucket_names.get(
        "root",
        GALILEO_DEFAULT_RUN_BUCKET_NAME,
    )
    config.results_bucket_name = _bucket_names.get(
        "results",
        GALILEO_DEFAULT_RESULT_BUCKET_NAME,
    )
    config.images_bucket_name = _bucket_names.get(
        "images",
        GALILEO_DEFAULT_IMG_BUCKET_NAME,
    )
    config.minio_fqdn = _dq_healthcheck_response.get(
        "minio_fqdn", os.getenv("MINIO_FQDN", None)
    )
    if config.minio_fqdn is not None and config.minio_fqdn.endswith(
        EXOSCALE_FQDN_SUFFIX
    ):
        config.is_exoscale_cluster = True

    proj_created_str = "new" if proj_created else "existing"
    run_created_str = "new" if run_created else "existing"
    print(
        f"🛰 Connected to {proj_created_str} project '{project_name}', "
        f"and {run_created_str} run '{run_name}'."
    )

    config.update_file_config()
    if config.current_project_id and config.current_run_id:
        _init.create_log_file_dir(
            config.current_project_id,
            config.current_run_id,
            overwrite_local=overwrite_local,
        )
    # Reset all config variables
    dataquality.get_data_logger().logger_config.reset(factory=True)
