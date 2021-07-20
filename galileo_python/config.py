import os
from typing import Optional


class Config:
    DEFAULT_NAME = ".galileo"

    def __init__(self, abs_dir_path: Optional[str] = None):
        self.abs_dir_path = abs_dir_path or f"{os.getcwd()}/{self.DEFAULT_NAME}"
        if not os.path.exists(self.abs_dir_path):
            os.makedirs(self.abs_dir_path)
