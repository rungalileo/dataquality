import json
import os
from enum import Enum, unique
from typing import Optional

from pydantic import BaseModel
from pydantic.types import UUID4, SecretStr


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
    api_url: str = os.getenv("GALILEO_API_URL") or "https://api.rungalileo.io"
    minio_url: str = os.getenv("GALILEO_MINIO_URL") or "https://minio.rungalileo.io"
    minio_access_key: str = "minioadmin"
    minio_secret_key: SecretStr = SecretStr("minioadmin")
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[SecretStr] = None
    current_user: Optional[str] = None
    current_project_id: Optional[UUID4] = None
    current_run_id: Optional[UUID4] = None

    def update_file_config(self) -> None:
        _config = _Config()
        _config.write_config(self.json())


config = Config()
if os.path.exists(_Config.DEFAULT_GALILEO_CONFIG_FILE):
    with open(_Config.DEFAULT_GALILEO_CONFIG_FILE) as f:
        config = Config(**json.load(f))
