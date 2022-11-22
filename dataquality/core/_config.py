import json
import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

import requests
from packaging import version
from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.types import UUID4
from requests.exceptions import ConnectionError as ReqConnectionError

from dataquality import __version__ as dq_version
from dataquality.exceptions import GalileoException
from dataquality.schemas.route import Route
from dataquality.schemas.task_type import TaskType
from dataquality.utils.helpers import galileo_disabled

CLOUD_URL = "https://console.cloud.rungalileo.io"
MINIMUM_API_VERSION = "0.4.0"


class GalileoConfigVars(str, Enum):
    API_URL = "GALILEO_API_URL"
    CONSOLE_URL = "GALILEO_CONSOLE_URL"

    @staticmethod
    def get_config_mapping() -> Dict[str, Optional[str]]:
        return {i.name.lower(): os.environ.get(i.value) for i in GalileoConfigVars}

    @staticmethod
    def get_available_config_attrs() -> Dict[str, str]:
        return {
            i.name.lower(): os.environ.get(i.value, "")
            for i in GalileoConfigVars
            if os.environ.get(i.value)
        }

    @staticmethod
    def auto_init_vars_available() -> bool:
        return bool(os.getenv("GALILEO_API_URL"))


class ConfigData(str, Enum):
    DEFAULT_GALILEO_CONFIG_DIR = f"{os.environ.get('HOME', str(Path.home()))}/.galileo"
    DEFAULT_GALILEO_CONFIG_FILE = f"{DEFAULT_GALILEO_CONFIG_DIR}/config.json"


class Config(BaseModel):
    api_url: str
    token: Optional[str] = None
    current_user: Optional[str] = None
    current_project_id: Optional[UUID4] = None
    current_run_id: Optional[UUID4] = None
    task_type: Optional[TaskType] = None

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def update_file_config(self) -> None:
        config_json = self.dict()

        with open(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value, "w+") as f:
            f.write(json.dumps(config_json, default=str))

    @validator("api_url", pre=True, always=True, allow_reuse=True)
    def add_scheme(cls, v: str) -> str:
        if not v.startswith("http"):
            # api url needs the scheme
            v = f"http://{v}"
        return v


def url_is_localhost(url: str) -> bool:
    return any(["localhost" in url, "127.0.0.1" in url])


def set_platform_urls(console_url_str: str) -> None:
    if url_is_localhost(console_url_str):
        os.environ[GalileoConfigVars.API_URL] = "http://localhost:8088"
    else:
        # some urls are set up as "console-xxx.domain instead of console.xxx.domain
        sfx = "." if "console." in console_url_str else "-"
        api_url = console_url_str.replace(f"console{sfx}", f"api{sfx}").rstrip("/")
        api_url = f"https://{api_url}" if not api_url.startswith("http") else api_url
        api_url = api_url.replace("http://", "https://")
        _validate_api_url(console_url_str, api_url)
        os.environ[GalileoConfigVars.API_URL] = api_url


def _validate_api_url(console_url: str, api_url: str) -> None:
    """Ensures the api url is a valid one"""
    err_detail = (
        f"The provided console URL {console_url} is invalid or is not currently "
        "available. If you are sure this is correct, reach out to your admin.\n\n"
        "To change your console url, run: \n`os.environ['GALILEO_CONSOLE_URL']='URL'` "
        "and then \n`dq.configure()`"
        "\n\nDetail: {err}"
    )
    url = f"https://{api_url}" if not api_url.startswith("http") else api_url
    try:
        r = requests.get(f"{url}/{Route.healthcheck}")
        if not r.ok:
            raise GalileoException(err_detail.format(err=r.text)) from None
    except (ReqConnectionError, ConnectionError) as e:
        raise GalileoException(err_detail.format(err=str(e))) from None


def _check_dq_version() -> None:
    """Check that user is running valid version of DQ client
    Pings backend to check minimum DQ version requirements.
    """
    r = requests.get(f"{config.api_url}/{Route.healthcheck}/dq")
    if not r.ok:
        if r.status_code == 404:
            # We don't want to raise error if api doesn't have dq healthcheck yet
            return
        raise GalileoException(r.text) from None

    dq_version_parsed = version.parse(dq_version)

    min_dq_version = r.json()["minimum_dq_version"]
    if dq_version_parsed < version.parse(min_dq_version):
        msg = (
            f"⚠️ You are running an old version of dataquality. Please upgrade to "
            f"version {min_dq_version} or higher (you are running {dq_version})."
            f"  `pip install dataquality --upgrade`"
        )
        # The user is running an incompatible DQ version, must upgrade
        raise GalileoException(msg)

    # If user is running an old API with new incompatible DQ version,
    # prompt them to downgrade
    api_version = r.json()["api_version"]
    if version.parse(api_version) < version.parse(MINIMUM_API_VERSION):
        msg_version = (
            f"{dq_version_parsed.major}.{dq_version_parsed.minor}"  # type: ignore
        )
        msg = (
            "Your Galileo API version is out of date. Please downgrade dataquality to "
            f'`pip install "dataquality<{msg_version}"` or contact your admin to '
            "upgrade your Galileo account."
        )
        raise GalileoException(msg)


def _check_console_url() -> None:
    """Checks for user setting of GALILEO_CONSOLE_URL instead of

    GALILEO_API_URL. If set, this will automatically set
    platform urls (GALILEO_API_URL) for auto_init
    """
    console_url = os.getenv(GalileoConfigVars.CONSOLE_URL)
    if console_url:
        if (
            "console." not in console_url
            and "console-" not in console_url
            and not url_is_localhost(console_url)
        ):
            warnings.warn(
                f"It seems your GALILEO_CONSOLE_URL ({console_url}) is invalid. "
                f"Your console URL should have 'console.' in the url. Ignoring"
            )
        else:
            set_platform_urls(console_url_str=console_url)


def set_config(cloud: bool = True) -> Config:
    if galileo_disabled():
        return Config(api_url="")
    _check_console_url()
    if not os.path.isdir(ConfigData.DEFAULT_GALILEO_CONFIG_DIR.value):
        os.makedirs(ConfigData.DEFAULT_GALILEO_CONFIG_DIR.value, exist_ok=True)
    if os.path.exists(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value):
        with open(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value) as f:
            try:
                config_vars: Dict[str, str] = json.load(f)
            # If there's an issue reading the config file for any reason, quit and
            # start fresh
            except Exception as e:
                warnings.warn(
                    f"We had an issue reading your config file ({type(e)}). "
                    f"Recreating your file from scratch."
                )
                return reset_config()
        # If the user updated any config vars via env, grab those updates
        new_config_attrs = GalileoConfigVars.get_available_config_attrs()
        config_vars.update(**new_config_attrs)
        config = Config(**config_vars)

    elif GalileoConfigVars.auto_init_vars_available():
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)

    else:
        name = "Galileo Cloud" if cloud else "Galileo"
        print(f"Welcome to {name} {dq_version}!")
        if cloud:
            console_url = CLOUD_URL
        else:
            print(
                "To skip this prompt in the future, set the following environment "
                "variable: GALILEO_CONSOLE_URL"
            )
            console_url = input("🔭 Enter the url of your Galileo console\n")
        set_platform_urls(console_url_str=console_url)
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)
    config.update_file_config()
    return config


def reset_config(cloud: bool = True) -> Config:
    """Wipe the config file and reconfigure"""
    if os.path.isfile(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value):
        os.remove(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value)
    return set_config(cloud)


config = set_config()
