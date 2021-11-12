import json
import os
from enum import Enum, unique
from getpass import getpass
from typing import Dict, List, Optional

from pydantic import BaseModel
from pydantic.types import UUID4, StrictStr


class GalileoConfigVars(str, Enum):
    API_URL = "GALILEO_API_URL"
    MINIO_URL = "GALILEO_MINIO_URL"
    MINIO_ACCESS_KEY = "GALILEO_MINIO_ACCESS_KEY"
    MINIO_SECRET_KEY = "GALILEO_MINIO_SECRET_KEY"
    MINIO_REGION = "GALILEO_MINIO_REGION"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, GalileoConfigVars))

    @staticmethod
    def get_config_mapping() -> Dict[str, str]:
        return {i.name.lower(): os.environ[i.value] for i in GalileoConfigVars}

    @staticmethod
    def vars_available() -> bool:
        return all(os.getenv(i) for i in GalileoConfigVars.get_valid_attributes())


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
    minio_access_key: str
    minio_secret_key: str
    minio_region: str
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[str] = None
    current_user: Optional[str] = None
    current_project_id: Optional[UUID4] = None
    current_run_id: Optional[UUID4] = None
    labels: Optional[List[StrictStr]] = None
    observed_num_labels: Optional[int] = 0

    class Config:
        validate_assignment = True

    def update_file_config(self) -> None:
        _config = _Config()
        _config.write_config(self.json())


if os.path.exists(_Config.DEFAULT_GALILEO_CONFIG_FILE):
    with open(_Config.DEFAULT_GALILEO_CONFIG_FILE) as f:
        config = Config(**json.load(f))

else:
    if not GalileoConfigVars.vars_available():
        print("Welcome to Galileo! To get started, we need some information:")
        print(
            "(To skip this prompt in the future, set the following environment "
            f"variables: {GalileoConfigVars.get_valid_attributes()})"
        )
        console_url = input("🔭 Enter the url of your Galileo console\n")
        print(f"you entered {console_url}")
        api_url = console_url.replace("console.", "api.")
        minio_url = console_url.replace("console.", "data.")
        if console_url.startswith("http"):
            # Minio url cannot have the scheme in the url
            minio_url = minio_url.split("://")[-1]
        else:
            # api url needs the scheme
            api_url = f"http://{api_url}"

        os.environ[GalileoConfigVars.API_URL] = api_url
        os.environ[GalileoConfigVars.MINIO_URL] = minio_url
        os.environ[GalileoConfigVars.MINIO_ACCESS_KEY] = input(
            "🔑 Enter the access key of your Galileo Minio server\n"
        )
        os.environ[GalileoConfigVars.MINIO_SECRET_KEY] = getpass(
            "🤫 Enter the secret key of your Galileo Minio server\n"
        )
        os.environ[GalileoConfigVars.MINIO_REGION] = (
            input(
                "📍 Enter the region of your Galileo Minio server [default us-east-1]\n"
            )
            or "us-east-1"
        )
    galileo_vars = GalileoConfigVars.get_config_mapping()
    config = Config(**galileo_vars)
