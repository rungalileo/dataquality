from typing import List, Optional, Union, Tuple

import numpy as np

from dataquality.loggers.logger_config.seq2seq.encoder_decoder import (
    EncoderDecoderLoggerConfig,
    encoder_decoder_logger_config,
)
from dataquality.loggers.model_logger.seq2seq.seq2seq_base import Seq2SeqModelLogger
from dataquality.utils.seq2seq import remove_padding
from dataquality.utils.seq2seq.logprobs import get_top_logprob_indices


class EncoderDecoderModelLogger(Seq2SeqModelLogger):
    """Seq2Seq model logger for EncoderDecoder models

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    __logger_name__ = "seq2seq" # "encoder_decoder"
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

        (
            self.token_logprobs,
            self.top_logprobs,
        ) = self.process_logprobs(
            self.ids, self.logits  # type: ignore
        )

    def format_sample(
            self, sample_id: int, sample_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Formats sample_logprobs and sample_top_indices

        TODO Comment

        Removes padding.

        Returns:
            - formatted_labels: np.ndarray
            - formatted_sample_logprobs: np.ndarray
            - formatted_sample_top_indices: np.ndarray
        """
        sample_n_tokens = sample_logits.shape[0]
        # TODO this could be abstracted away
        sample_labels = self._retrieve_sample_labels(
            sample_id, max_tokens=sample_n_tokens
        )
        padding_side = self.logger_config.tokenizer.padding_side
        num_sample_tokens = len(sample_labels)
        sample_logits = remove_padding(
            sample_logits,
            num_sample_tokens,
            padding_side,
        )

        sample_logprobs = self.convert_logits_to_logprobs(sample_logits)
        sample_top_indices = get_top_logprob_indices(sample_logprobs)

        return sample_labels, sample_logprobs, sample_top_indices
