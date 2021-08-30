import os
from typing import Optional

import jsonlines
import numpy as np
from pydantic.types import UUID4

from dataquality.core.config import _Config
from dataquality.utils.hdf5_store import HDF5Store


class JsonlLogger:
    INPUT_FILENAME = "input_data.jsonl"
    OUTPUT_FILENAME = "model_output_data.jsonl"
    EMB_LOG_FILENAME = "model_output_embeddings.h5"
    LOG_FILE_DIR = f"{_Config.DEFAULT_GALILEO_CONFIG_DIR}/logs"

    def __init__(self) -> None:
        self.hdf5_store: Optional[HDF5Store] = None
        self.log_file_dir = f"{self.LOG_FILE_DIR}"
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

    def write_output(
        self, project_id: UUID4, run_id: UUID4, data: dict, emb_column_name: str = "emb"
    ) -> None:
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

        # grab the embeddings
        emb = np.array(data[emb_column_name])
        if (
            not self.hdf5_store
            or self.hdf5_store.datapath != f"{write_output_dir}/{self.EMB_LOG_FILENAME}"
        ):
            self.hdf5_store = HDF5Store(
                f"{write_output_dir}/{self.EMB_LOG_FILENAME}",
                emb_column_name,
                shape=emb.shape,
            )
        self.hdf5_store.write(emb)

        # swap embeddings for the id
        data[emb_column_name] = self.hdf5_store.i
        output_writer.write(data)
