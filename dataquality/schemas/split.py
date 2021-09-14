from enum import Enum, unique
from typing import List


@unique
class Split(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"

    @staticmethod
    def get_valid() -> List[str]:
        return [Split.training, Split.validation, Split.test, Split.inference]
