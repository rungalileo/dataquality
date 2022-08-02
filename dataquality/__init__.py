"dataquality"

__version__ = "v0.3.8-alpha0"

import os
import resource

import dataquality.core._config
import dataquality.metrics
from dataquality.core._config import config
from dataquality.core.auth import login, verify_jwt_token
from dataquality.core.finish import finish, get_run_status, wait_for_run
from dataquality.core.init import init
from dataquality.core.log import (
    docs,
    get_data_logger,
    get_model_logger,
    log_data_sample,
    log_data_samples,
    log_dataset,
    log_model_outputs,
    set_epoch,
    set_labels_for_run,
    set_split,
    set_tagging_schema,
    set_tasks_for_run,
)
from dataquality.utils.dq_logger import get_dq_log_file
from dataquality.utils.helpers import check_noop


@check_noop
def configure() -> None:
    """Update your active config with new env variables.

    Available environment variables to update:
    * GALILEO_CONSOLE_URL
    """
    if "GALILEO_API_URL" in os.environ:
        del os.environ["GALILEO_API_URL"]
    updated_config = dataquality.core._config.reset_config()
    for k, v in updated_config.dict().items():
        config.__setattr__(k, v)
    config.token = os.getenv("GALILEO_JWT_TOKEN")
    config.update_file_config()
    verify_jwt_token()


__all__ = [
    "__version__",
    "init",
    "log_data_samples",
    "log_model_outputs",
    "login",
    "config",
    "configure",
    "finish",
    "set_labels_for_run",
    "get_data_logger",
    "get_model_logger",
    "set_tasks_for_run",
    "set_tagging_schema",
    "docs",
    "wait_for_run",
    "get_run_status",
    "set_epoch",
    "set_split",
    "log_data_sample",
    "log_dataset",
    "get_dq_log_file",
]

try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
except ValueError:  # The users limit is higher than our max, which is OK
    pass
