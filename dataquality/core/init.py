from typing import Optional

from dataquality.utils.name import random_name


class _Init:
    def __init__(self, project: Optional[str] = None):
        self.project = project or f"{random_name()}-{random_name()}"


def init() -> None:
    _init = _Init()
    print(_init)
