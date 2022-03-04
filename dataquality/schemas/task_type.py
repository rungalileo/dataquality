from enum import Enum, unique
from typing import List


@unique
class TaskType(str, Enum):
    """Valid task types supported for logging by Galileo"""

    text_classification = "text_classification"
    text_multi_label = "text_multi_label"
    text_ner = "text_ner"

    @staticmethod
    def get_valid_tasks() -> List[str]:
        return list(map(lambda x: x.value, TaskType))
