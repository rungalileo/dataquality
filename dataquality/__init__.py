"dataquality"

__version__ = "v0.8.13"

import os
import warnings

import dataquality.core._config
import dataquality.integrations

# We try/catch this in case the user installed dq inside of jupyter. You need to
# restart the kernel after the install and we want to make that clear. This is because
# of vaex: https://github.com/vaexio/vaex/pull/2226
from dataquality.exceptions import GalileoWarning
from dataquality.schemas.model import ModelFramework
from dataquality.schemas.split import Split

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
    log_image_dataset,
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
    warnings.warn(
        "configure is deprecated, use dq.set_console_url and dq.login", GalileoWarning
    )

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
    "log_image_dataset",
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


class DataQuality:
    call_args = None
    call_kwargs = None

    def __init__(self, *args, **kwargs):
        self.call_kwargs = kwargs
        self.call_args = args
        print("INIT")

    def __call__(self, *args, **kwds):
        self.call_kwargs = kwds
        self.call_args = args
        print("CALL")
        return self

    def setup_dq(self, model, args, kwargs, framework):
        task_type = kwargs.pop("task", kwargs.pop("task_type", None))
        run_name = kwargs.pop("run", kwargs.pop("run_name", None))
        project_name = kwargs.pop("project", kwargs.pop("project_name", None))
        labels = kwargs.pop("labels", [])
        train_df = kwargs.pop("train_df", None)
        test_df = kwargs.pop("test_df", None)
        val_df = kwargs.pop("val_df", None)

        assert (
            task_type is not None
        ), """keyword argument task_type is required,
for example task_type='classification' """

        assert (
            labels is not None
        ), """keyword labels is required,
for example task_type=['neg','pos']"""
        init_kwargs = {
            "task_type": task_type,
        }
        if run_name:
            init_kwargs["run_name"] = run_name
        if project_name:
            init_kwargs["project_name"] = project_name
        init(**init_kwargs)
        if framework == ModelFramework.spacy:
            from dataquality.integrations.spacy import log_input_examples
            from dataquality.integrations.spacy import watch as spacy_watch

            model.initialize(lambda: train_df)
            spacy_watch(model)
            if train_df is not None:
                log_input_examples(train_df, split=Split.train)
            if test_df is not None:
                log_input_examples(test_df, split=Split.test)
            if val_df is not None:
                log_input_examples(val_df, split=Split.validation)
        else:
            if labels:
                set_labels_for_run(labels)
            if train_df is not None:
                log_dataset(train_df, split=Split.train)
            if test_df is not None:
                log_dataset(test_df, split=Split.test)
            if val_df is not None:
                log_dataset(val_df, split=Split.validation)

    def guess_framework(self, model):
        if hasattr(model, "pipe"):
            return ModelFramework.spacy
        elif hasattr(model, "fit"):
            return ModelFramework.keras
        elif hasattr(model, "register_forward_hook"):
            return ModelFramework.torch
        elif hasattr(model, "push_to_hub"):
            return ModelFramework.hf

    def start_watching(self, args, kwargs):
        model = kwargs.get("model")
        if not model:
            model = args[0]
        self.model = model
        framework = kwargs.get("framework", self.guess_framework(model))
        self.setup_dq(model, args, kwargs, framework)
        # We want to support the following models
        if framework == ModelFramework.spacy:  # spacy
            from dataquality.integrations.spacy import unwatch as spacy_unwatch

            self.unwatch = spacy_unwatch
        elif framework == ModelFramework.keras:
            from dataquality.integrations.experimental.keras import (
                unwatch as keras_unwatch,
            )
            from dataquality.integrations.experimental.keras import watch as keras_watch

            self.unwatch = keras_unwatch
            keras_watch(model)
        elif framework == ModelFramework.hf:
            from dataquality.integrations.transformers_trainer import (
                unwatch as trainer_unwatch,
            )
            from dataquality.integrations.transformers_trainer import (
                watch as trainer_watch,
            )

            self.unwatch = trainer_unwatch
            trainer_watch(model)
        elif framework == ModelFramework.torch:
            from dataquality.integrations.torch import unwatch as torch_unwatch
            from dataquality.integrations.torch import watch as torch_watch

            self.unwatch = torch_unwatch
            torch_watch(model)
        else:
            print("Model could not be determined")
            # raise GalileoException("model class could not be determined")
        return self

    def __enter__(self, *args, **kwargs):
        print("ENTER")
        if not len(args) and not len(kwargs):
            args = self.call_args
            kwargs = self.call_kwargs
        self.start_watching(args, kwargs)
        return self

    def __exit__(self, *args, **kwargs):
        if hasattr(self, "unwatch"):
            self.unwatch(self.model)
        finish()
        return self

    # we want to add the __all__ to the module
    def __dir__(self):
        return __all__

    def __getattr__(self, name):
        if name in __all__:
            return globals()[name]
        raise AttributeError(f"module {__name__} has no attribute {name}")


# Workaround by Guido van Rossum:
# https://mail.python.org/pipermail/python-ideas/2012-May/014969.html
# This allows us to use the same syntax as the original dataquality package
# _dq = DataQuality()
# sys.modules[__name__] = _dq
