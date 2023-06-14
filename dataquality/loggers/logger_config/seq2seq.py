from typing import Dict, Any

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class Seq2SeqLoggerConfig(BaseLoggerConfig):
    sample_length: Dict[str, int] = {}
    tokenizer: Any = None



seq2seq_logger_config = Seq2SeqLoggerConfig()
