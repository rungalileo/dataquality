from dataquality.loggers.data_logger.seq2seq.seq2seq_base import Seq2SeqDataLogger
from dataquality.loggers.logger_config.seq2seq.chat import (
    Seq2SeqChatLoggerConfig,
    seq2seq_chat_logger_config,
)


class Seq2SeqChatDataLogger(Seq2SeqDataLogger):
    __logger_name__ = "seq2seq_chat"
    logger_config: Seq2SeqChatLoggerConfig = seq2seq_chat_logger_config
