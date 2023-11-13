from collections import defaultdict
from typing import Dict, List, Optional, Set

from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig
from dataquality.schemas.seq2seq import Seq2SeqModelTypes
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
    model_type: Seq2SeqModelTypes = Seq2SeqModelTypes.encoder_decoder
    # Decoder only below
    id_to_formatted_prompt_length: Dict[str, Dict[int, int]] = defaultdict(dict)
    response_template: Optional[List[int]] = None

    # @validator("response_template")
    # def ensure_tokenized_response_template(
    #     cls, response_template: Optional[Union[str, List[int]]], values: Dict[str, Any]  # noqa
    # ) -> List[int]:
    #     if response_template and isinstance(response_template, str):
    #         tokenizer = values.get("tokenizer")
    #         if tokenizer:
    #             response: List[int] = tokenizer.encode(
    #                 response_template, padding=False, add_special_tokens=False
    #             )
    #             return response

    #     return response_template or []

    class Config:
        arbitrary_types_allowed = True


seq2seq_logger_config = Seq2SeqLoggerConfig()
