from enum import Enum


class JobName(str, Enum):
    """
    List of available jobs to run on data from the python client.
    """

    default = "default"
