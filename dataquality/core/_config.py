import json
import os
import warnings
from enum import Enum, unique
from typing import Dict, Optional

from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.types import UUID4

from dataquality.schemas.task_type import TaskType


class GalileoConfigVars(str, Enum):
    API_URL = "GALILEO_API_URL"
    MINIO_URL = "GALILEO_MINIO_URL"
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
        return bool(os.getenv("GALILEO_MINIO_URL") and os.getenv("GALILEO_API_URL"))


class ConfigData(str, Enum):
    DEFAULT_GALILEO_CONFIG_DIR = f"{os.getcwd()}/.galileo"
    DEFAULT_GALILEO_CONFIG_FILE = f"{DEFAULT_GALILEO_CONFIG_DIR}/config.json"
    minio_secret_key = "_minio_secret_key"


@unique
class AuthMethod(str, Enum):
    email = "email"


class Config(BaseModel):
    api_url: str
    minio_url: str
    minio_region: str = "us-east-1"
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[str] = None
    current_user: Optional[str] = None
    current_project_id: Optional[UUID4] = None
    current_run_id: Optional[UUID4] = None
    task_type: Optional[TaskType] = None
    _minio_secret_key: str = ""

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def update_file_config(self) -> None:
        config_json = self.dict()
        config_json[ConfigData.minio_secret_key.value] = self._minio_secret_key

        with open(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value, "w+") as f:
            f.write(json.dumps(config_json, default=str))

    @validator("minio_url", pre=True, always=True, allow_reuse=True)
    def remove_scheme(cls, v: str) -> str:
        if v.startswith("http"):
            # Minio url cannot have the scheme - fqdm
            v = v.split("://")[-1]
        return v

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
        os.environ[GalileoConfigVars.MINIO_URL] = "http://localhost:9000"
    else:
        os.environ[GalileoConfigVars.API_URL] = console_url_str.replace(
            "console.", "api."
        )
        os.environ[GalileoConfigVars.MINIO_URL] = console_url_str.replace(
            "console.", "data."
        )


def _check_console_url() -> None:
    """Checks for user setting of GALILEO_CONSOLE_URL instead of

    GALILEO_API_URL and GALILEO_MINIO_URL. If set, this will automatically set
    platform urls (GALILEO_API_URL and GALILEO_MINIO_URL) for auto_init
    """
    console_url = os.getenv(GalileoConfigVars.CONSOLE_URL)
    if console_url:
        if "console." not in console_url and not url_is_localhost(console_url):
            warnings.warn(
                f"It seems your GALILEO_CONSOLE_URL ({console_url}) is invalid. "
                f"Your console URL should have 'console.' in the url. Ignoring"
            )
        else:
            set_platform_urls(console_url_str=console_url)


def set_config() -> Config:
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
                os.remove(ConfigData.DEFAULT_GALILEO_CONFIG_FILE.value)
                return set_config()
        # If the user updated any config vars via env, grab those updates
        new_config_attrs = GalileoConfigVars.get_available_config_attrs()
        config_vars.update(**new_config_attrs)
        config = Config(**config_vars)
        # Need to set private pydantic fields explicitly
        config._minio_secret_key = config_vars.get(
            ConfigData.minio_secret_key.value, ""
        )

    elif GalileoConfigVars.auto_init_vars_available():
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)

    else:
        print("Welcome to Galileo!")
        print(
            "To skip this prompt in the future, set the following environment "
            "variable: GALILEO_CONSOLE_URL"
        )
        console_url = input("🔭 Enter the url of your Galileo console\n")
        set_platform_urls(console_url_str=console_url)
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)
    return config


config = set_config()
config.update_file_config()
