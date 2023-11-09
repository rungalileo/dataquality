from typing import Dict, Union, List, Any

from pydantic import validator

from mypy.binder import defaultdict

from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig


class DecoderOnlyLoggerConfig(Seq2SeqLoggerConfig):
    """Decoder-Only logger config

    Adds:
        - Id_to_full_sample_length (used for removing padding)
    """
    # For each split/inference-name, store sample id -> int as the
    # length of full tokenized prompt
    id_to_formatted_prompt_length: Dict[str, Dict[int, int]] = defaultdict(dict)
    # Template indicating the start of the target output (i.e. assistant response)
    response_template: Union[str, List[int]] = None

    @validator("response_template")
    def ensure_tokenized_response_template(
        cls, response_template: Union[str, List[int]], values: Dict[str, Any]
    ) -> List[int]:
        if isinstance(response_template, str):
            tokenizer = values.get("tokenizer")
            return tokenizer.encode(
                response_template,
                padding=False,
                add_special_tokens=False
            )

        return response_template


decoder_only_logger_config = DecoderOnlyLoggerConfig()
