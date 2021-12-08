"dataquality"

__version__ = "0.0.4"

import resource

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
)

__all__ = [
    "__version__",
    "login",
    "init",
    "log_input_data",
    "log_batch_input_data",
    "log_model_outputs",
    "config",
    "finish",
    "set_labels_for_run",
    "get_data_logger",
    "get_model_logger",
]

resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
