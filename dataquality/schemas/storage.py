from enum import Enum


class Serialization(str, Enum):
    """
    List of available serialization formats
    """

    pickle = "pkl"
    jsonl = "jsonl"
