import json
import os
from enum import Enum, unique
from typing import Dict, Optional

from pydantic import BaseModel


class _Config:
    DEFAULT_GALILEO_CONFIG_FILE = f"{os.getcwd()}/.galileo/config.json"

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

    def write_config(self, data: Dict) -> None:
        with open(self.config_file, "w+") as f:
            f.write(json.dumps(data))

    def config(self) -> "Config":
        return Config(**self.config_dict)


@unique
class AuthMethod(str, Enum):
    email = "email"


class Config(BaseModel):
    api_url: str = os.getenv("GALILEO_API_URL") or "https://api.rungalileo.io"
    auth_method: AuthMethod = AuthMethod.email
    token: Optional[str] = None
    current_user: Optional[str] = None
    current_project: Optional[str] = None
    current_run: Optional[str] = None

    def update_file_config(self) -> None:
        _config = _Config()
        _config.write_config(self.dict())


config = Config()
if os.path.exists(_Config.DEFAULT_GALILEO_CONFIG_FILE):
    with open(_Config.DEFAULT_GALILEO_CONFIG_FILE) as f:
        config = Config(**json.load(f))
