from collections import defaultdict
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizerFast

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class Seq2SeqLoggerConfig(BaseLoggerConfig):
    sample_length: Dict[str, int] = {}
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    split_token_map: Dict[str, Dict[int, List[int]]] = defaultdict(dict)

    class Config:
        arbitrary_types_allowed = True


seq2seq_logger_config = Seq2SeqLoggerConfig()
