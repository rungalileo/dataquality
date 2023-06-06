from enum import Enum
from enum import unique


@unique
class ModelFramework(str, Enum):
    torch = "torch"
    keras = "keras"
    hf = "hf"
    spacy = "spacy"
    auto = "auto"
