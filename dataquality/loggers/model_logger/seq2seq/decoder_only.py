from typing import List, Optional, Union, Tuple

import numpy as np

from dataquality.loggers.logger_config.seq2seq.decoder_only import DecoderOnlyLoggerConfig, decoder_only_logger_config
from dataquality.loggers.logger_config.seq2seq.encoder_decoder import (
    EncoderDecoderLoggerConfig,
    encoder_decoder_logger_config,
)
from dataquality.loggers.model_logger.seq2seq.seq2seq_base import Seq2SeqModelLogger
from dataquality.utils.seq2seq import remove_padding
from dataquality.utils.seq2seq.logprobs import get_top_logprob_indices


class EncoderDecoderModelLogger(Seq2SeqModelLogger):
    """Seq2Seq model logger for EncoderDecoder models

    TODO Update!

    Since Encoder-Decoder models output logits just over the target tokens,
    there is very little additional processing - i.e. we primarily leverage
    functionality from Seq2SeqModelLogger.
    """

    __logger_name__ = "seq2seq"
    logger_config: DecoderOnlyLoggerConfig = decoder_only_logger_config
    log_file_ext = "arrow"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        #logprobs: Optional[Union[List, np.ndarray]] = None,  # TODO Add this for people to directly log liklihoods
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
            #logprobs=logprobs,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
            labels=labels,
        )

    def _retrieve_num_sample_tokens(self, sample_id: int, max_tokens: int) -> Tuple[int, Optional[int]]:
        """Retrieves the number of tokens in the formatted_prompt

        This is used to remove padding tokens before restricting to the response tokens.

        If the num_sample_tokens > max_tokens - meaning the user tokenized their
        data with a smaller max_length, for not we should throw an error!

        What this really means is we need to keep track of how many tokens to strip
        from the response labels. Take this e.g.
            - num_sample_tokens = 135
            - max_tokens = 128
            - num_sample_tokens - max_tokens = 7
            - We need to remove the last 7 tokens of the response labels.
            sample_labels = sample_labels[:-7]

        So to handle this, we should return num_sample_tokens and the difference
        if we have max_tokens!
        """
        num_sample_tokens = self.logger_config.id_to_formatted_prompt_length[self.token_map_key][sample_id]

        if num_sample_tokens > max_tokens:
            return max_tokens, num_sample_tokens - max_tokens

        return num_sample_tokens, None

    def format_sample(
        self, sample_id: int, sample_tokens: np.ndarray, shift_labels: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formats sample_logprobs and sample_top_indices

        TODO comment!
            Note that sample_tokens can be logits OR logprobs

        Removes padding and (for DecoderOnly models) restricts to just
        response tokens.

        Returns:
            - formatted_labels: np.ndarray
            - formatted_logits: np.ndarray
        """
        sample_n_tokens = sample_tokens.shape[0]
        num_sample_labels, num_extra_tokens = self._retrieve_num_sample_tokens(sample_id, sample_n_tokens)

        response_labels = self._retrieve_sample_labels(
            sample_id, max_tokens=None
        )
        # TODO Check this logic - especially around getting the correct
        #   sample size!
        if num_extra_tokens:
            response_labels = response_labels[:-num_extra_tokens]

        padding_side = self.logger_config.tokenizer.padding_side

        sample_wo_padding = remove_padding(
            sample_tokens,
            num_sample_labels,
            padding_side
        )

        # Add a flag to shift or not!
        # Restrict to just the response tokens
        num_response_tokens = len(response_labels)
        # TODO check - Shift the logits such that tokens < n predict token n.
        #   notice here that we ignore the final token logprob since there is
        #   no n+1 token. For DecoderOnly the logits and labels are implicitly
        #   shifted within the model.
        # WHEN LOGGING LOGPROBS DIRECTLY THIS MAY MESS  THINGS UP!
        if shift_labels:
            sample_response = sample_wo_padding[-(num_response_tokens + 1): -1]
        else:
            sample_response = sample_wo_padding[-num_response_tokens:]

        return response_labels, sample_response

