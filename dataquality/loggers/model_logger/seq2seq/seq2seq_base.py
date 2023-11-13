from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from dataquality.loggers.logger_config.seq2seq.seq2seq_base import (
    Seq2SeqLoggerConfig,
    seq2seq_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.loggers.model_logger.seq2seq.formatters import get_model_formatter
from dataquality.schemas.seq2seq import TOP_LOGPROBS_SCHEMA
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file
from dataquality.utils.seq2seq.logprobs import (
    process_sample_logprobs,
)


class Seq2SeqModelLogger(BaseGalileoModelLogger):
    """Seq2Seq base model logger

    This class defines the base functionality for logging model outputs in
    Seq2Seq tasks - shared between EncoderDecoder and DecoderOnly architectures.

    After architecture specific processing of raw model logits, we leverage
    a shared function for processing and extracting the logprob token data **just**
    over the Target data.

    During processing, the following key information is extracted:
        - token_logprobs: log-probs for GT tokens in each sample
        - top_logprobs: top-K (token_str, log-prob) pairs for each token
    """

    __logger_name__ = "seq2seq"
    logger_config: Seq2SeqLoggerConfig = seq2seq_logger_config
    log_file_ext = "arrow"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        logprobs: Optional[Union[List, np.ndarray]] = None,
        ids: Optional[Union[List, np.ndarray]] = None,
        split: str = "",
        epoch: Optional[int] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            embs=embs,
            probs=probs,
            logits=logits,
            ids=ids,
            split=split,
            epoch=epoch,
            inference_name=inference_name,
        )
        self.logprobs = logprobs if logprobs is not None else []
        self.token_logprobs = pa.array([])
        self.top_logprobs = pa.array([])
        # Formatter distinguishes behavior between EncoderDecoder and DecoderOnly
        model_type = self.logger_config.model_type
        split_key = (
            str(self.split)
            if (self.split != Split.inference and self.inference_name is not None)
            else str(self.inference_name)
        )
        self.formatter = get_model_formatter(model_type, self.logger_config, split_key)

    def validate_and_format(self) -> None:
        """Validate the lengths, calculate token level dep, extract GT probs"""
        self.logits = self._convert_tensor_ndarray(self.logits)
        self.logprobs = self._convert_tensor_ndarray(self.logprobs)
        self.ids = self._convert_tensor_ndarray(self.ids)
        assert len(self.ids) == len(self.logits), (
            "Must pass in a valid batch with equal id and logit length, got "
            f"id: {len(self.ids)},logits: {len(self.logits)}"
        )

        assert (
            self.logger_config.tokenizer is not None
        ), "Must set your tokenizer. Use `dq.integrations.seq2seq.hf.set_tokenizer`"
        (
            self.token_logprobs,
            self.top_logprobs,
        ) = self.process_logprobs(
            self.ids, self.logits  # type: ignore
        )

    def process_logprobs(
        self, batch_ids: np.ndarray, batch_logits: np.ndarray
    ) -> Tuple[pa.array, pa.array]:
        """Handle processing for a batch of sample logits

        For each sample in the batch extract / compute the following values:
            - Token level logprobs for the GT label
            - Token level top-k model logprobs: represented as a dictionary
            mapping {predicted_token: logprob}

        batch_logits has shape - [batch_size, max_token_length, vocab_size], where
        max_token_length is determined by the longest sample in the batch. Because
        other samples in the batch are padded to this max_length, we have to process
        each sample individually to ignore pad token indices.

        Special points of consideration:
            - For each sample, top-k logprobs is a list of dictionaries with length
            equal to the number of tokens in that sample. Each dictionary maps the
            models top-k predicted tokens to their corresponding logprobs.

            - We return a pyarrow array because each sample may have a different number
            of token, which can't be represented in numpy.

        Returns:
            batch_token_logprobs: GT Logprob per token
                len(batch_token_dep) == batch_size
                batch_token_logprobs[i].shape is [num_tokens_in_label[i]]
            batch_top_logprobs: Top-k logprob dictionary per token
                type(batch_top_logprobs[i]) = List[Dict[str, float]]
                len(batch_top_logprobs) == batch_size
                len(batch_top_logprobs[i]) = num_tokens_in_label[i]
        """
        assert self.logger_config.tokenizer is not None  # Needed for linting

        batch_token_logprobs = []
        batch_top_logprobs = []
        # Iterate through the samples in the batch
        for sample_id, sample_logits in zip(batch_ids, batch_logits):
            (
                sample_labels,
                sample_logprobs,
                sample_top_indices,
            ) = self.formatter.format_sample(sample_id, sample_logits)

            logprob_data = process_sample_logprobs(
                sample_logprobs=sample_logprobs,
                sample_labels=sample_labels,
                sample_top_indices=sample_top_indices,
                tokenizer=self.logger_config.tokenizer,
            )
            batch_token_logprobs.append(logprob_data.token_logprobs)
            batch_top_logprobs.append(logprob_data.top_logprobs)

        return (
            pa.array(batch_token_logprobs),
            pa.array(batch_top_logprobs, type=TOP_LOGPROBS_SCHEMA),
        )

    def _get_data_dict(self) -> Dict:
        """Returns the data dictionary for writing to disk"""
        batch_size = len(self.ids)
        data = {
            C.id.value: self.ids,
            C.token_logprobs.value: self.token_logprobs,
            C.top_logprobs.value: self.top_logprobs,
            C.split_.value: [Split[self.split].value] * batch_size,
            C.epoch.value: [self.epoch] * batch_size,
        }
        if self.split == Split.inference:
            data[C.inference_name.value] = [self.inference_name] * batch_size
        return data

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        save_arrow_file(path, object_name, data)
