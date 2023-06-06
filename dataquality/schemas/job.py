from enum import Enum
from enum import unique


@unique
class JobName(str, Enum):
    default = "default"
    inference = "inference"
