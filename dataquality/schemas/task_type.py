from enum import Enum, unique
from typing import List


@unique
class TaskType(str, Enum):
    """Valid task types supported for logging by Galileo"""

    text_classification = "text_classification"
    text_multi_label = "text_multi_label"
    text_ner = "text_ner"
    image_classification = "image_classification"
    tabular_classification = "tabular_classification"
    object_detection = "object_detection"
    semantic_segmentation = "semantic_segmentation"

    @staticmethod
    def get_valid_tasks() -> List[str]:
        return list(map(lambda x: x.value, TaskType))

    @staticmethod
    def get_mapping(task_int: int) -> "TaskType":
        """Converts the servers task type enum to client names"""
        return {
            0: TaskType.text_classification,
            1: TaskType.text_multi_label,
            2: TaskType.text_ner,
            3: TaskType.image_classification,
            4: TaskType.tabular_classification,
            5: TaskType.object_detection,
            6: TaskType.semantic_segmentation,
        }[task_int]
