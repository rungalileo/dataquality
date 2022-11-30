"dataquality"

__version__ = "v0.8.3a10"

import os
import resource

import dataquality.core._config
import dataquality.integrations

# We try/catch this in case the user installed dq inside of jupyter. You need to
# restart the kernel after the install and we want to make that clear. This is because
# of vaex: https://github.com/vaexio/vaex/pull/2226
try:
    import dataquality.metrics
    from dataquality.analytics import Analytics
    from dataquality.clients.api import ApiClient
except (FileNotFoundError, AttributeError):
    raise Exception(
        "It looks like you've installed dataquality from a notebook. "
        "Please restart the kernel before continuing"
    ) from None
from dataquality.core._config import config
from dataquality.core.auth import login, logout
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
    set_epoch_and_split,
    set_labels_for_run,
    set_split,
    set_tagging_schema,
    set_tasks_for_run,
)
from dataquality.core.report import build_run_report, register_run_report
from dataquality.dq_auto.auto import auto
from dataquality.schemas.condition import (
    AggregateFunction,
    Condition,
    ConditionFilter,
    Operator,
)
from dataquality.utils.dq_logger import get_dq_log_file
from dataquality.utils.helpers import (
    check_noop,
    disable_galileo,
    disable_galileo_verbose,
    enable_galileo,
    enable_galileo_verbose,
)


@check_noop
def configure(do_login: bool = True) -> None:
    """[Not for cloud users] Update your active config with new information

    You can use environment variables to set the config, or wait for prompts
    Available environment variables to update:
    * GALILEO_CONSOLE_URL
    * GALILEO_USERNAME
    * GALILEO_PASSWORD
    """
    a.log_function("dq/configure")

    if "GALILEO_API_URL" in os.environ:
        del os.environ["GALILEO_API_URL"]
    updated_config = dataquality.core._config.reset_config(cloud=False)
    for k, v in updated_config.dict().items():
        config.__setattr__(k, v)
    config.token = None
    config.update_file_config()
    if do_login:
        login()


@check_noop
def set_console_url(console_url: str = None) -> None:
    """For Enterprise users. Set the console URL to your Galileo Environment.

    You can also set GALILEO_CONSOLE_URL before importing dataquality to bypass this

    :param console_url: If set, that will be used. Otherwise, if an environment variable
    GALILEO_CONSOLE_URL is set, that will be used. Otherwise, you will be prompted for
    a url.
    """
    a.log_function("dq/set_console_url")
    if console_url:
        os.environ["GALILEO_CONSOLE_URL"] = console_url
    configure(do_login=False)


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
]

try:
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
except ValueError:  # The users limit is higher than our max, which is OK
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
