from dataquality.loggers.logger_config.seq2seq.chat import (
    Seq2SeqChatLoggerConfig,
    seq2seq_chat_logger_config,
)
from dataquality.loggers.model_logger.seq2seq.seq2seq_base import Seq2SeqModelLogger


class Seq2SeqChatModelLogger(Seq2SeqModelLogger):
    __logger_name__ = "seq2seq_chat"
    logger_config: Seq2SeqChatLoggerConfig = seq2seq_chat_logger_config
