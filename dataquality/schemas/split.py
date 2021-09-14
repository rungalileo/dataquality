from enum import Enum, unique
from typing import List


@unique
class Split(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, Split))
