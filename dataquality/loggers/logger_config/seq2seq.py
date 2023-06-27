from collections import defaultdict
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizerFast

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class Seq2SeqLoggerConfig(BaseLoggerConfig):
    sample_length: Dict[str, int] = {}
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    # For each split/inference-name, store sample id -> List[token_id] for the label
    id_to_tokens: Dict[str, Dict[int, List[int]]] = defaultdict(dict)

    class Config:
        arbitrary_types_allowed = True


seq2seq_logger_config = Seq2SeqLoggerConfig()
