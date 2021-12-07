import os
import shutil
import warnings
from abc import abstractmethod
from glob import glob
from typing import TypeVar, Type

import numpy as np

from dataquality import config
from dataquality.loggers.config.base_config import (
    BaseGalileoConfig,
    BaseConfigAttributes
)

T = TypeVar('T', bound='BaseGalileoDataConfig')


class BaseGalileoDataConfig(BaseGalileoConfig):
    MAX_META_COLS = 50  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 50  # Max characters in a string metadata attribute
    INPUT_DATA_NAME = "input_data.arrow"

    def __init__(self, **kwargs):
        super().__init__()
        self.is_data_config = True
        self.is_model_config = False
        self.meta = {}

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def log(self):
        pass

    def upload(self):
        pass

    def validate_metadata(self, batch_size: int):
        if len(self.meta.keys()) > self.MAX_META_COLS:
            warnings.warn(
                f"You can only log up to {self.MAX_META_COLS} metadata attrs. "
                f"The first {self.MAX_META_COLS} will be logged only."
            )
        # When logging metadata columns, if the user breaks a rule, don't fail
        # completely, just warn them and remove that metadata column
        # Cast to list for in-place dictionary mutation
        reserved_keys = BaseConfigAttributes.get_valid()
        valid_meta = {}
        for key, values in list(self.meta.items())[:self.MAX_META_COLS]:
            # Key must not override a default
            if key in reserved_keys:
                warnings.warn(
                    f"Metadata column names must not override default values "
                    f"{reserved_keys}. This metadata field "
                    f"will be removed."
                )
                continue
            # Must be the same length as input
            if len(values) != batch_size:
                warnings.warn(
                    f"Expected {batch_size} values for key {key} but got "
                    f"{len(values)}. Will not log this metadata column."
                )
                continue
            # Values must be a point, not an iterable
            valid_types = (str, int, float, np.floating, np.integer)
            invalid_values = filter(
                lambda t: not isinstance(t, valid_types)
                or (isinstance(t, str) and len(t) > self.MAX_STR_LEN),
                values,
            )
            bad_val = next(invalid_values, None)
            if bad_val:
                warnings.warn(
                    f"Metadata column {key} has one or more invalid values {bad_val} "
                    f"of type {type(bad_val)}. Only strings of len < {self.MAX_STR_LEN} "
                    "and numbers can be logged."
                )
                continue
            valid_meta[key] = values
        self.meta = valid_meta

    def get_config(self, task_type: str) -> Type['BaseGalileoDataConfig']:
        super().get_config(task_type)
        configs = {i.__name__: i for i in BaseGalileoDataConfig.__subclasses__()}
        print('HERE')
        print(configs)
        print(BaseGalileoDataConfig.__subclasses__())
        return configs[task_type]

    def _cleanup(self) -> None:
        """
        Cleans up the current run data locally
        """
        assert config.current_project_id
        assert config.current_run_id
        location = (
            f"{self.LOG_FILE_DIR}/{config.current_project_id}"
            f"/{config.current_run_id}"
        )
        print("🧹 Cleaning up")
        for path in glob(f"{location}/*"):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
