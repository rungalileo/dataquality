from typing import Optional

from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.schemas.task_type import TaskType


def get_task_type(task_type: Optional[TaskType] = None) -> TaskType:
    task = task_type or config.task_type
    if not task:
        raise GalileoException(
            "You must provide either a task_type or first call "
            "dataqualtiy.init and provide one"
        )
    return task
