from dataquality.loggers.logger_config.seq2seq.seq2seq_base import Seq2SeqLoggerConfig


class EncoderDecoderLoggerConfig(Seq2SeqLoggerConfig):
    """Encoder Decoder logger config

    For now, the Encoder Decoder logger config has the same fields
    as the base Seq2Seq logger config
    """


encoder_decoder_logger_config = EncoderDecoderLoggerConfig()
