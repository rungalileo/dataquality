import os

import jsonlines
from pydantic.types import UUID4

from dataquality.core.config import _Config


class JsonlLogger:
    INPUT_FILENAME = "input_data.jsonl"
    OUTPUT_FILENAME = "model_output_data.jsonl"

    def __init__(self) -> None:
        self.log_file_dir = f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs"
        if not os.path.exists(self.log_file_dir):
            os.makedirs(self.log_file_dir)

    def write_input(self, project_id: UUID4, run_id: UUID4, data: dict) -> None:
        write_input_dir = f"{self.log_file_dir}/{project_id}/{run_id}"
        if not os.path.exists(write_input_dir):
            os.makedirs(write_input_dir)
        input_writer = jsonlines.Writer(
            open(
                f"{write_input_dir}/{self.INPUT_FILENAME}",
                "a",
            ),
            flush=True,
        )
        input_writer.write(data)

    def write_output(self, project_id: UUID4, run_id: UUID4, data: dict) -> None:
        write_output_dir = f"{self.log_file_dir}/{project_id}/{run_id}"
        if not os.path.exists(write_output_dir):
            os.makedirs(write_output_dir)
        output_writer = jsonlines.Writer(
            open(
                f"{write_output_dir}/{self.OUTPUT_FILENAME}",
                "a",
            ),
            flush=True,
        )
        output_writer.write(data)
