from enum import Enum


class Pipeline(str, Enum):
    """
    List of available API pipelines
    """

    default = "default"
    calculate_metrics = "simple_jsonl_io_with_metrics"
