import json
import os
from enum import Enum, unique
from typing import Dict, List, Optional

from pydantic import BaseModel
from pydantic.class_validators import validator
from pydantic.types import UUID4

from dataquality.schemas.task_type import TaskType


class GalileoConfigVars(str, Enum):
    API_URL = "GALILEO_API_URL"
    MINIO_URL = "GALILEO_MINIO_URL"
    MINIO_ACCESS_KEY = "MINIO_ACCESS_KEY"
    MINIO_SECRET_KEY = "MINIO_SECRET_KEY"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, GalileoConfigVars))

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
        return all(os.getenv(i) for i in ["GALILEO_API_URL", "GALILEO_MINIO_URL"])


class _Config:
    DEFAULT_GALILEO_CONFIG_DIR = f"{os.getcwd()}/.galileo"
    DEFAULT_GALILEO_CONFIG_FILE = f"{DEFAULT_GALILEO_CONFIG_DIR}/config.json"

    def __init__(self) -> None:
        self.config_file = self.DEFAULT_GALILEO_CONFIG_FILE
        self._setup_config_dir()
        self._setup_config_file()
        self.config_dict = self._load_config_file()

    def _setup_config_dir(self) -> None:
        if not os.path.exists(self.config_file):
            dirname = os.path.dirname(self.config_file)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

    def _setup_config_file(self) -> None:
        if not os.path.exists(self.config_file):
            with open(self.config_file, "w+") as f:
                f.write("{}")

    def _load_config_file(self) -> dict:
        with open(self.config_file) as f:
            return json.load(f)

    def write_config(self, data: str) -> None:
        with open(self.config_file, "w+") as f:
            f.write(data)

    def config(self) -> "Config":
        return Config(**self.config_dict)


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

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

    def update_file_config(self) -> None:
        _config = _Config()
        _config.write_config(self.json())

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


def set_platform_urls(console_url_str: str) -> None:
    if console_url_str == "localhost":
        os.environ[GalileoConfigVars.API_URL] = "http://localhost:8088"
        os.environ[GalileoConfigVars.MINIO_URL] = "http://localhost:9000"
    else:
        os.environ[GalileoConfigVars.API_URL] = console_url_str.replace(
            "console.", "api."
        )
        os.environ[GalileoConfigVars.MINIO_URL] = console_url_str.replace(
            "console.", "data."
        )


def set_config() -> Config:
    if os.path.exists(_Config.DEFAULT_GALILEO_CONFIG_FILE):
        with open(_Config.DEFAULT_GALILEO_CONFIG_FILE) as f:
            config_vars: Dict[str, str] = json.load(f)
        # If the user updated any config vars via env, grab those updates
        new_config_attrs = GalileoConfigVars.get_available_config_attrs()
        config_vars.update(**new_config_attrs)
        config = Config(**config_vars)

    elif GalileoConfigVars.auto_init_vars_available():
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)

    else:
        print("Welcome to Galileo!")
        print(
            "To skip this prompt in the future, set the following environment "
            "variables: GALILEO_API_URL GALILEO_MINIO_URL"
        )
        console_url = input("🔭 Enter the url of your Galileo console\n")
        set_platform_urls(console_url_str=console_url)
        galileo_vars = GalileoConfigVars.get_config_mapping()
        config = Config(**galileo_vars)
    return config


config = set_config()
config.update_file_config()
