"dataquality"

__version__ = "v0.0.5"

import resource

import dataquality.core._config
from dataquality.core._config import config
from dataquality.core.auth import login
from dataquality.core.finish import finish
from dataquality.core.init import init
from dataquality.core.log import (
    get_data_logger,
    get_model_logger,
    log_batch_input_data,
    log_input_data,
    log_model_outputs,
    set_labels_for_run,
    set_tasks_for_run,
)


def configure() -> None:
    """Update your active config with new env variables.

    Available environment variables to update:
    * GALILEO_API_URL
    * GALILEO_MINIO_URL
    """
    updated_config = dataquality.core._config.set_config()
    for k, v in updated_config.dict().items():
        config.__setattr__(k, v)
    config.update_file_config()


__all__ = [
    "__version__",
    "login",
    "init",
    "log_input_data",
    "log_batch_input_data",
    "log_model_outputs",
    "config",
    "configure",
    "finish",
    "set_labels_for_run",
    "get_data_logger",
    "get_model_logger",
    "set_tasks_for_run",
]

resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
