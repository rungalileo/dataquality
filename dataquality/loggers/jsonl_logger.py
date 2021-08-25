import os

import jsonlines

from dataquality.core.config import Config, _Config


class JsonlLogger:
    LOG_FILE_NAME = "data.jsonl"

    def __init__(self, config: Config) -> None:
        self.log_file_dir = (
            f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs/"
            f"{config.current_project_id}/{config.current_run_id}"
        )
        self.log_file_path = f"{self.log_file_dir}/{self.LOG_FILE_NAME}"
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)
        self.writer = jsonlines.Writer(open(self.log_file_path, "w"), flush=True)
