from collections import defaultdict
from typing import Dict, Set, Union

from dataquality.loggers.logger_config.text_classification import (
    TextClassificationLoggerConfig,
)


class ImageClassificationLoggerConfig(TextClassificationLoggerConfig):
    # Keep track of the ids that have been observed in the current epoch
    # the key is the split and epoch like observed_ids["train_0"] = {0, 1, 2, 3}
    observed_ids: Dict[str, Set[Union[str, int]]] = defaultdict(set)


image_classification_logger_config = ImageClassificationLoggerConfig()
