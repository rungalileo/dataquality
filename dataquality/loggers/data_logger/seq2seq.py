from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.loggers.data_logger import BaseGalileoDataLogger


class Seq2SeqDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "seq2seq"
    logger_config = seq2seq_logger_config
