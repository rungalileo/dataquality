import os
from typing import Dict, Optional

from dataquality.schemas.sdk_config import SDKConfig


class _Config:
    DEFAULT_NAME = ".galileo"

    def __init__(self, abs_dir_path: Optional[str] = None):
        self.abs_dir_path = abs_dir_path or f"{os.getcwd()}/{self.DEFAULT_NAME}"
        if not os.path.exists(self.abs_dir_path):
            os.makedirs(self.abs_dir_path)


def config(sdk_config: Optional[Dict] = None) -> None:
    _config = _Config()
    _sdk_config = SDKConfig()
    if sdk_config:
        _sdk_config = SDKConfig(**sdk_config)
    print(_config)
    print(_sdk_config)
