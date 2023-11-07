from dataquality.loggers.logger_config.seq2seq.completion import (
    Seq2SeqCompletionLoggerConfig,
    seq2seq_completion_logger_config,
)
from dataquality.loggers.model_logger.seq2seq.seq2seq_base import Seq2SeqModelLogger


class Seq2SeqCompletionModelLogger(Seq2SeqModelLogger):
    __logger_name__ = "seq2seq_completion"
    logger_config: Seq2SeqCompletionLoggerConfig = seq2seq_completion_logger_config
