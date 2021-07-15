from enum import Enum, unique

from pydantic import BaseModel


@unique
class LoggerMode(str, Enum):
    training = "training"
    validation = "validation"
    test = "test"
    inference = "inference"

class InputLogItem(BaseModel):
    pass

class OutputLogItem(BaseModel):
    pass
