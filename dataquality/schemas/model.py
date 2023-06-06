from enum import Enum, unique


@unique
class ModelFramework(str, Enum):
    torch = "torch"
    keras = "keras"
    hf = "hf"
    spacy = "spacy"
    auto = "auto"
