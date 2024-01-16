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
    prompt_evaluation = "prompt_evaluation"
    seq2seq = "seq2seq"  # soon to be deprecated in favor or chat or completion
    llm_monitor = "llm_monitor"
    seq2seq_completion = "seq2seq_completion"
    seq2seq_chat = "seq2seq_chat"

    @staticmethod
    def get_valid_tasks() -> List["TaskType"]:
        """Tasks that are valid for dataquality."""
        return [
            task_type
            for task_type in TaskType
            if task_type not in [TaskType.prompt_evaluation, TaskType.llm_monitor]
        ]

    @staticmethod
    def get_seq2seq_tasks() -> List["TaskType"]:
        """Sequence to Sequence tasks types."""
        return [
            TaskType.seq2seq,
            TaskType.seq2seq_completion,
            TaskType.seq2seq_chat,
        ]

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
            7: TaskType.prompt_evaluation,
            8: TaskType.seq2seq,  # soon to be deprecated in favor or chat or completion
            9: TaskType.llm_monitor,
            10: TaskType.seq2seq_completion,
            11: TaskType.seq2seq_chat,
        }[task_int]
