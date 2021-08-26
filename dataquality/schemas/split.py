from enum import Enum, unique


@unique
class Split(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"
