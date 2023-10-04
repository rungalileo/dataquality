from typing import List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.seq2seq.encoder_decoder import (
    EncoderDecoderLoggerConfig,
    encoder_decoder_logger_config,
)
from dataquality.loggers.model_logger.seq2seq.seq2seq import Seq2SeqModelLogger


class EncoderDecoderModelLogger(Seq2SeqModelLogger):
    # TODO Add in API so we can use encoder_decoder
    # __logger_name__ = "encoder_decoder"
    __logger_name__ = "seq2seq"
    logger_config: EncoderDecoderLoggerConfig = encoder_decoder_logger_config
    log_file_ext = "arrow"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
            labels=labels,
        )

    def validate_and_format(self) -> None:
        """Compute token level log-prob info for Encoder-Decoder Models

        Encoder-Decoder models output `logits` just over the target tokens.
        Therefore, we can very easily extract token log-prob info without
        any additional data formatting / token splitting.
        """
        super().validate_and_format()

        # TODO: [JON] computing softmax on GPU can lead to speedups of ~5x
        # TODO: Question, the validation done in the parent class does not seem
        #   to propigate. Here e.g. we convert ids to np.array in super()
        logprobs = self.convert_logits_to_logprobs(self.logits)
        (
            self.token_logprobs,
            self.top_logprobs,
        ) = self.process_logprobs(
            self.ids, logprobs  # type: ignore
        )
