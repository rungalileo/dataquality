import warnings
from abc import abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

import numpy as np

from dataquality.loggers.base_logger import BaseGalileoLogger, BaseLoggerAttributes

T = TypeVar("T", bound="BaseGalileoDataLogger")


class BaseGalileoDataLogger(BaseGalileoLogger):
    MAX_META_COLS = 50  # Limit the number of metadata attrs a user can log
    MAX_STR_LEN = 50  # Max characters in a string metadata attribute
    INPUT_DATA_NAME = "input_data.arrow"

    def __init__(self, meta: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.meta: Dict[str, Any] = meta or {}

    @abstractmethod
    def validate(self) -> None:
        pass

    @abstractmethod
    def log(self) -> None:
        pass

    @classmethod
    def upload(cls) -> None:
        pass

    def validate_metadata(self, batch_size: int) -> None:
        if len(self.meta.keys()) > self.MAX_META_COLS:
            warnings.warn(
                f"You can only log up to {self.MAX_META_COLS} metadata attrs. "
                f"The first {self.MAX_META_COLS} will be logged only."
            )
        # When logging metadata columns, if the user breaks a rule, don't fail
        # completely, just warn them and remove that metadata column
        # Cast to list for in-place dictionary mutation
        reserved_keys = BaseLoggerAttributes.get_valid()
        valid_meta = {}
        for key, values in list(self.meta.items())[: self.MAX_META_COLS]:
            # Key must not override a default
            if key in reserved_keys:
                warnings.warn(
                    f"Metadata column names must not override default values "
                    f"{reserved_keys}. Metadata field {key} "
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
                    f"of type {type(bad_val)}. Only strings of "
                    f"len < {self.MAX_STR_LEN} and numbers can be logged."
                )
                continue
            valid_meta[key] = values
        self.meta = valid_meta

    @staticmethod
    def get_logger(task_type: str) -> Type["BaseGalileoDataLogger"]:
        BaseGalileoLogger.validate_task(task_type)
        loggers = {i.__logger_name__: i for i in BaseGalileoDataLogger.__subclasses__()}
        return loggers[task_type]

    @staticmethod
    def get_data_logger_attr(cls: object) -> str:
        """
        Returns the attribute that corresponds to the GalileoDataConfig class.
        This assumes only 1 GalileoDataConfig object exists in the class

        :param cls: The class
        :return: The attribute name
        """
        for attr in dir(cls):
            member_class = getattr(cls, attr)
            if isinstance(member_class, BaseGalileoDataLogger):
                return attr
        raise AttributeError("No GalileoDataConfig attribute found!")
