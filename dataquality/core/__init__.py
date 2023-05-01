import os
import warnings
from typing import Optional

import dataquality
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.exceptions import GalileoWarning
from dataquality.utils.helpers import check_noop

a = Analytics(ApiClient, config)


@check_noop
def configure(do_login: bool = True, _internal: bool = False) -> None:
    """[Not for cloud users] Update your active config with new information

    You can use environment variables to set the config, or wait for prompts
    Available environment variables to update:
    * GALILEO_CONSOLE_URL
    * GALILEO_USERNAME
    * GALILEO_PASSWORD
    """
    a.log_function("dq/configure")
    if not _internal:
        warnings.warn(
            "configure is deprecated, use dq.set_console_url and dq.login",
            GalileoWarning,
        )

    if "GALILEO_API_URL" in os.environ:
        del os.environ["GALILEO_API_URL"]
    updated_config = dataquality.core._config.reset_config(cloud=False)
    for k, v in updated_config.dict().items():
        config.__setattr__(k, v)
    config.token = None
    config.update_file_config()
    if do_login:
        dataquality.core.auth.login()


@check_noop
def set_console_url(console_url: Optional[str] = None) -> None:
    """For Enterprise users. Set the console URL to your Galileo Environment.

    You can also set GALILEO_CONSOLE_URL before importing dataquality to bypass this

    :param console_url: If set, that will be used. Otherwise, if an environment variable
    GALILEO_CONSOLE_URL is set, that will be used. Otherwise, you will be prompted for
    a url.
    """
    a.log_function("dq/set_console_url")
    if console_url:
        os.environ["GALILEO_CONSOLE_URL"] = console_url
    configure(do_login=False, _internal=True)
