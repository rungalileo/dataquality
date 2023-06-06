from enum import Enum, unique


@unique
class JobName(str, Enum):
    default = "default"
    inference = "inference"
