from collections import defaultdict
from typing import Dict, Set, Union

from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class ImageClassificationLoggerConfig(TextClassificationLoggerConfig):
    observed_ids: Dict[str, Set[Union[str, int]]] = defaultdict(set)


image_classification_logger_config = ImageClassificationLoggerConfig()
