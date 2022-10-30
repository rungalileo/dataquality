from enum import Enum
from typing import List, Union

from dataquality.exceptions import GalileoException


class Split(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, Split))


def conform_split(split: Union[str, Split]) -> Split:
    """Conforms split name to our naming conventions

    Raises GalileoException if split is invalid
    """
    if isinstance(split, Split):
        return split
    if split == "train":  # Needed since HF datasets uses "train"
        return Split.training
    try:
        return Split[split]
    except KeyError:
        raise GalileoException(
            f"Split must be one of {Split.get_valid_attributes()} " f"but got {split}"
        )
