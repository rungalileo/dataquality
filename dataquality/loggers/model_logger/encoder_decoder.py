from typing import List, Optional, Union

import numpy as np

from dataquality.loggers.logger_config.encoder_decoder import EncoderDecoderLoggerConfig, encoder_decoder_logger_config
from dataquality.loggers.model_logger.seq2seq import Seq2SeqModelLogger


class EncoderDecoderModelLogger(Seq2SeqModelLogger):
    # TODO Add class level comment
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
        # TODO Update Comment
        """Validate through super() then calculate token level logprob information"""
        super().validate_and_format()

        # TODO: This is potentially slow. This is what needs to be optimized. Can we
        #  potentially do this on the GPU with torch? And dont convert to a np array
        #  [JON] computing softmax on GPU can lead to speedups of around 5x in my
        #  experience
        logprobs = self.convert_logits_to_logprobs(self.logits)
        (
            self.token_logprobs,
            self.top_logprobs,
        ) = self.process_logprobs(self.ids, logprobs)

