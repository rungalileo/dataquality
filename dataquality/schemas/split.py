from enum import Enum
from typing import List


class Split(str, Enum):
    training = "training"
    train = "training"
    validation = "validation"
    test = "test"
    testing = "test"
    inference = "inference"

    @staticmethod
    def get_valid_attributes() -> List[str]:
        return list(map(lambda x: x.value, Split))
