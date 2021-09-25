import os
import threading

import jsonlines
from pydantic.types import UUID4

from dataquality.core.config import _Config
from dataquality.schemas import Split

lock = threading.Lock()


class JsonlLogger:
    INPUT_FILENAME_SUFFIX = "input_data.jsonl"
    OUTPUT_FILENAME_SUFFIX = "model_output_data.jsonl"
    LOG_FILE_DIR = f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs"

    def __init__(self) -> None:
        self.log_file_dir = f"{self.LOG_FILE_DIR}"
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

    def write_input(
        self, split: Split, project_id: UUID4, run_id: UUID4, data: dict
    ) -> None:
        write_input_dir = f"{self.log_file_dir}/{project_id}/{run_id}"
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)
        with lock:
            with open(
                f"{write_input_dir}/{split}_{self.INPUT_FILENAME_SUFFIX}", "a"
            ) as f:
                input_writer = jsonlines.Writer(f, flush=True)
                input_writer.write(data)

    def write_output(
        self, split: Split, project_id: UUID4, run_id: UUID4, data: dict
    ) -> None:
        write_output_dir = f"{self.log_file_dir}/{project_id}/{run_id}"
        if not os.path.exists(write_output_dir):
            os.makedirs(write_output_dir)
        with lock:
            with open(
                f"{write_output_dir}/{split}_{self.OUTPUT_FILENAME_SUFFIX}", "a"
            ) as f:
                output_writer = jsonlines.Writer(f, flush=True)
                output_writer.write(data)
