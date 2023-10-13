from collections import defaultdict
from typing import Dict, List, Optional, Set

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig
from dataquality.schemas.split import Split


class Seq2SeqLoggerConfig(BaseLoggerConfig):
    sample_length: Dict[str, int] = {}
    tokenizer: Optional[PreTrainedTokenizerFast] = None
    max_input_tokens: Optional[int] = None
    max_target_tokens: Optional[int] = None
    # For each split/inference-name, store sample id -> List[token_id] for the label
    id_to_tokens: Dict[str, Dict[int, List[int]]] = defaultdict(dict)
    model: Optional[PreTrainedModel] = None
    generation_config: Optional[GenerationConfig] = None
    generation_splits: Set[Split] = set()

    class Config:
        arbitrary_types_allowed = True


seq2seq_logger_config = Seq2SeqLoggerConfig()
