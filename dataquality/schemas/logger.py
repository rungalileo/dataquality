from enum import Enum, unique


@unique
class LoggerMode(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"
