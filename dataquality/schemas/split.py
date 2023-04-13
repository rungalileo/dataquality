from enum import Enum
from typing import List, Union

from dataquality.exceptions import GalileoException


class Split(str, Enum):
    train = "training"
    training = "training"
    val = "validation"
    valid = "validation"
    validation = "validation"
    test = "test"
    testing = "test"
    inference = "inference"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, Split))

    @staticmethod
    def get_valid_keys() -> List[str]:
        return ["train", "training", "val", "valid", "validation", "test", "testing"]


def conform_split(split: Union[str, Split]) -> Split:
    """Conforms split name to our naming conventions

    Raises GalileoException if split is invalid
    """
    if isinstance(split, Split):
        return split
    try:
        return Split[split]
    except KeyError:
        raise GalileoException(
            f"Split must be one of {Split.get_valid_attributes()} " f"but got {split}"
        )
