from enum import Enum


class ProcName(str, Enum):
    """
    List of available procedures to run on data from the python client.
    """

    default = "default"
