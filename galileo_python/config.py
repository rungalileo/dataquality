import os
from typing import Optional


class Config:
    """Galileo Config.

    Contains logic and dictation for where any config and state should be
    managed for a Galileo object.
    """

    CONFIG_DIR_NAME = "galileo"

    def __init__(self, galileo_dir_path: Optional[str] = None):
        self.galileo_dir_path = (
            galileo_dir_path or f"{os.getcwd()}/.{self.CONFIG_DIR_NAME}"
        )
        if not os.path.exists(self.galileo_dir_path):
            os.makedirs(self.galileo_dir_path)
