"""dataquality is a library for tracking and analyzing your machine learning models.
:param model: The model to inspect, if a string, it will be assumed to be auto
:param task: Task type for example "text_classification"
:param project: Project name
:param run: Run name
:param train_data: Training data
:param test_data: Optional test data
:param val_data: Optional: validation data
:param labels: The labels for the run
:param framework: The framework to use, if provided it will be used instead of
    inferring it from the model. For example, if you have a torch model, you
    can pass framework="torch". If you have a torch model, you can pass
    framework="torch"
:param args: Additional arguments
:param kwargs: Additional keyword arguments
.. code-block:: python
    import dataquality
    with dataquality(
        model,
        "text_classification",
        labels = ["neg", "pos"],
        train_data = train_data
    ):
        model.fit(train_data)
If you want to train without a model, you can use the auto framework:
.. code-block:: python
    import dataquality
    with dataquality(labels = ["neg", "pos"],
                     train_data = train_data):
        dataquality.get_insights()
"""


__version__ = "v0.9.1"

import sys
from typing import Any, List, Optional

import dataquality.core._config
import dataquality.integrations

# We try/catch this in case the user installed dq inside of jupyter. You need to
# restart the kernel after the install and we want to make that clear. This is because
try:
    import dataquality.metrics
    from dataquality.analytics import Analytics
    from dataquality.clients.api import ApiClient
except (FileNotFoundError, AttributeError):
    raise Exception(
        "It looks like you've installed dataquality from a notebook. "
        "Please restart the kernel before continuing"
    ) from None
from dataquality.core import configure, set_console_url
from dataquality.core._config import config
from dataquality.core.auth import login, logout
from dataquality.core.finish import finish, get_run_status, wait_for_run
from dataquality.core.init import init
from dataquality.core.log import (
    docs,
    get_current_run_labels,
    get_data_logger,
    get_model_logger,
    log_data_sample,
    log_data_samples,
    log_dataset,
    log_image_dataset,
    log_model_outputs,
    log_xgboost,
    set_epoch,
    set_epoch_and_split,
    set_labels_for_run,
    set_split,
    set_tagging_schema,
    set_tasks_for_run,
)
from dataquality.core.report import build_run_report, register_run_report
from dataquality.dq_auto.auto import auto
from dataquality.dq_auto.notebook import auto_notebook
from dataquality.dq_start import DataQuality
from dataquality.schemas.condition import (
    AggregateFunction,
    Condition,
    ConditionFilter,
    Operator,
)
from dataquality.utils.dq_logger import get_dq_log_file
from dataquality.utils.helpers import (
    disable_galileo,
    disable_galileo_verbose,
    enable_galileo,
    enable_galileo_verbose,
)

__all__ = [
    "__version__",
    "login",
    "logout",
    "init",
    "log_data_samples",
    "log_model_outputs",
    "config",
    "configure",
    "finish",
    "set_labels_for_run",
    "get_current_run_labels",
    "get_data_logger",
    "get_model_logger",
    "set_tasks_for_run",
    "set_tagging_schema",
    "docs",
    "wait_for_run",
    "get_run_status",
    "set_epoch",
    "set_split",
    "set_epoch_and_split",
    "set_console_url",
    "log_data_sample",
    "log_dataset",
    "log_image_dataset",
    "log_xgboost",
    "get_dq_log_file",
    "build_run_report",
    "register_run_report",
    "AggregateFunction",
    "Operator",
    "Condition",
    "ConditionFilter",
    "disable_galileo",
    "disable_galileo_verbose",
    "enable_galileo_verbose",
    "enable_galileo",
    "auto",
    "DataQuality",
    "auto_notebook",
]

try:
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
except (ImportError, ValueError):  # The users limit is higher than our max, which is OK
    pass

#  Logging is optional. If enabled, imports, method calls
#  and exceptions can be logged by calling the logger.
#  This is useful for debugging and detecting issues.
#  Logging is disabled by default for enterprise users.
#  To enable logging, set the environment variable
#  DQ_TELEMETRICS=1
#  To log initiate the Analytics class and pass in the gallileo ApiClient + dq.config
#  a = Analytics(ApiClient, config)
#  Once initialized you can start logging
#  a.log_import("dataquality")
#  a.log_method_call("dataquality.log_data_samples")
a = Analytics(ApiClient, config)
a.log_import("dataquality")


class _DataQuality:
    """This class is used to create a singleton instance of the DataQuality class.

    This is done to allow the user to use the same syntax as the original dataquality
    package. The original package had a singleton instance of the DataQuality class
    that was created when the package was imported. This class is used to mimic that
    behavior.
    """

    _instance: Optional[DataQuality] = None

    def __init__(self) -> None:
        self._instance = None

    def __call__(self, *args: Any, **kwargs: Any) -> DataQuality:
        """Return the singleton instance of the DataQuality class."""
        if self._instance is None:
            self._instance = DataQuality(*args, **kwargs)
        return self._instance

        # we want to add the __all__ to the module

    def get_insights(self) -> None:
        return

    def __dir__(self) -> List[str]:
        return __all__

    def __getattr__(self, name: str) -> Any:
        if name in __all__:
            return globals()[name]
        # We return the wanted import from the original dataquality package
        else:
            return getattr(dataquality, name)


# Workaround by Guido van Rossum:
# https://mail.python.org/pipermail/python-ideas/2012-May/014969.html
# This allows us to use the same syntax as the original dataquality package
sys.modules[__name__] = _DataQuality()  # type: ignore
