import json
import os
from typing import Dict, Optional

from dataquality.schemas.config import Config


class _Config:
    GALILEO_CONFIG_FILE = "GALILEO_CONFIG"
    DEFAULT_GALILEO_CONFIG_FILE = f"{os.getcwd()}/.galileo/config.json"

    def __init__(self) -> None:
        self.config_file = os.getenv(
            self.GALILEO_CONFIG_FILE,
            self.DEFAULT_GALILEO_CONFIG_FILE,
        )
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

    def config(self) -> Config:
        return Config(**self.config_dict)


def config(params: Optional[Dict] = {}) -> Config:
    _config = _Config()
    if params:
        _config = _Config()
        _config.write_config(params)
    else:
        _config = _Config()
        _config.write_config(Config().dict())
    return _config.config()
