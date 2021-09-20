from enum import Enum


class Pipeline(str, Enum):
    """
    List of available API pipelines
    """

    calculate_metrics = "simple_jsonl_io_with_metrics"
