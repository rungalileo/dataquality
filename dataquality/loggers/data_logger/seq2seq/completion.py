from dataquality.loggers.data_logger.seq2seq.seq2seq_base import Seq2SeqDataLogger
from dataquality.loggers.logger_config.seq2seq.completion import (
    Seq2SeqCompletionLoggerConfig,
    seq2seq_completion_logger_config,
)


class Seq2SeqCompletionDataLogger(Seq2SeqDataLogger):
    __logger_name__ = "seq2seq_completion"
    logger_config: Seq2SeqCompletionLoggerConfig = seq2seq_completion_logger_config
