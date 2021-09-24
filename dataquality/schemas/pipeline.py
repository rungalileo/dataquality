from enum import Enum


class Pipeline(str, Enum):
    """
    List of available API pipelines
    """

    default = "default"
