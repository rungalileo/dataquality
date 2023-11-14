from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from scipy.special import log_softmax

from dataquality.loggers.logger_config.seq2seq.seq2seq_base import (
    Seq2SeqLoggerConfig,
    seq2seq_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.seq2seq import TOP_LOGPROBS_SCHEMA, TOP_K
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file
from dataquality.utils.seq2seq import (
    remove_padding,
)
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
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

    logger_config: Seq2SeqLoggerConfig = seq2seq_logger_config
    log_file_ext = "arrow"

    def __init__(
        self,
        embs: Optional[Union[List, np.ndarray]] = None,
        probs: Optional[Union[List, np.ndarray]] = None,
        logits: Optional[Union[List, np.ndarray]] = None,
        #logprobs: Optional[Union[List, np.ndarray]] = None,
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
        #self.logprobs = logprobs if logprobs is not None else []
        self.token_logprobs = pa.array([])
        self.top_logprobs = pa.array([])
        self.labels = labels

    @property
    def token_map_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        """Validate shared data format for seq2seq

        Note that this base fn does the validation
        Sub classes that inherit from this can have modality specific
         formatting (such as EncoderDecoder and DecoderOnly)
        """
        if self.labels is not None:
            self.labels = self._convert_tensor_ndarray(self.labels)
        self.logits = self._convert_tensor_ndarray(self.logits)
        # self.logprobs = self._convert_tensor_ndarray(self.logprobs)
        self.probs = self._convert_tensor_ndarray(self.probs)
        self.ids = self._convert_tensor_ndarray(self.ids)
        # TODO Figure out what to do here
        # assert len(self.ids) == len(self.logits), (
        #     "Must pass in a valid batch with equal id and logit length, got "
        #     f"id: {len(self.ids)},logits: {len(self.logits)}"
        # )
        if self.labels is not None:
            assert len(self.labels) == len(self.ids), "TODO: Must be same len message"

        assert (
            self.logger_config.tokenizer is not None
        ), "Must set your tokenizer. Use `dq.integrations.seq2seq.hf.set_tokenizer`"

        if self.probs is not None:
            # TODO Fow now create a fake here!!
            (
                self.token_logprobs,
                self.top_logprobs
            ) = self.process_logprobs(self.ids, self.probs)
        else:
            (
                self.token_logprobs,
                self.top_logprobs,
            ) = self.process_logits(
                self.ids, self.logits  # type: ignore
            )

    def process_logprobs(
        self, batch_ids: np.ndarray, batch_logprobs: np.ndarray
    ) -> pa.array:
        """Format logprobs primarily by removing padding / restricting to responses"""
        batch_token_logprobs = []
        batch_top_logprobs = []
        for sample_id, sample_logprobs in zip(
                batch_ids, batch_logprobs
        ):
            # Note that with API based models the logprob data is
            # already shifted / aligned.
            response_labels, sample_response_logprobs = self.format_sample(
                sample_id,
                sample_logprobs
            )

            # Add fake top loprobs
            sample_top_logprobs: List[List[Tuple[str, float]]] = []
            for response_token_id, response_token_logprob in zip(response_labels, sample_response_logprobs):  # Loop over the tokens
                response_token_str = self.logger_config.tokenizer.decode(response_token_id)
                token_top_logprobs_mapping = [(response_token_str, response_token_logprob)] + [("---", -1e10)] * (TOP_K - 1)
                sample_top_logprobs.append(token_top_logprobs_mapping)

            batch_token_logprobs.append(sample_response_logprobs)
            batch_top_logprobs.append(sample_top_logprobs)

        return (
            pa.array(batch_token_logprobs),
            pa.array(batch_top_logprobs, type=TOP_LOGPROBS_SCHEMA),
        )

    @abstractmethod
    def format_sample(
            self, sample_id: int, sample_tokens: np.ndarray, shift_labels: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Formats and generates sample_logprobs and sample_top_indices

        TODO Fix comment!
            Note that sample_tokens can be logits OR logprobs

        First, we remove padding and (for DecoderOnly models) restricts to just
        response tokens.

        Then we convert to logprobs using `logsoftmax` and compute the top-k
        per token indices.

        NOTE: These operations are all done at the sample level, due to shape
        misalignment when removing padding etc. Moreover, we delay the computation
        of logprobs and top_indices to avoid wasted computation over padding and
        non-response tokens

        Returns:
            - formatted_labels: np.ndarray
            - formatted_logits: np.ndarray
        """
        pass

    def process_logits(
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
        for sample_id, sample_logits in zip(
                batch_ids, batch_logits
        ):
            sample_labels, sample_logits = self.format_sample(
                sample_id, sample_logits, shift_labels=True
            )

            sample_logprobs = self.convert_logits_to_logprobs(sample_logits)
            sample_top_indices = get_top_logprob_indices(sample_logprobs)

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
            C.split_.value: [Split[self.split].value] * batch_size,
            C.epoch.value: [self.epoch] * batch_size,
        }
        # If the user logged straight logprobs, we don't have access
        # to top_logprobs
        if len(self.top_logprobs) != 0:
            data[C.top_logprobs.value] = self.top_logprobs

        if self.split == Split.inference:
            data[C.inference_name.value] = [self.inference_name] * batch_size
        return data

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        save_arrow_file(path, object_name, data)

    def _retrieve_sample_labels(self, sample_id: int, max_tokens: Optional[int]) -> np.ndarray:
        """Retrieve the labels array based on the sample id and truncate at max_tokens

        Labels gives the ground truth / target sample ids for
        each token in the sequence:

        e.g. for sample_id = 8 --> labels = [0, 10, 16, ...]
        """
        labels = self.logger_config.id_to_tokens[self.token_map_key][sample_id]
        if max_tokens is not None:
            labels = labels[:max_tokens]
        return np.array(labels)

    def convert_logits_to_logprobs(
        self, sample_logits: Union[List[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Converts logits (unnormalized log probabilities) to logprobs via log_softmax

        This is a special use case for Seq2Seq, people generally
        work with logprobs. One reason for this is that the logsoftmax
        function takes advantage of the logsumexp "trick" to compute a
        numerically stable version of log(softmax(x)).
        """
        # axis ensures that in a matrix of probs with dims num_samples x num_classes
        # we take the softmax for each sample
        if not isinstance(sample_logits, np.ndarray):
            sample_logits = self._convert_tensor_ndarray(sample_logits)

        return log_softmax(sample_logits, axis=-1)
