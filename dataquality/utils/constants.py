import os
from enum import Enum
from pathlib import Path


class ConfigData(str, Enum):
    DEFAULT_GALILEO_CONFIG_DIR = f"{os.environ.get('HOME', str(Path.home()))}/.galileo"
    DEFAULT_GALILEO_CONFIG_FILE = f"{DEFAULT_GALILEO_CONFIG_DIR}/config.json"
