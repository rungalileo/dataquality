from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa

from dataquality.loggers.logger_config.seq2seq import seq2seq_logger_config
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.split import Split
from dataquality.utils.arrow import save_arrow_file


class Seq2SeqModelLogger(BaseGalileoModelLogger):
    __logger_name__ = "seq2seq"
    logger_config = seq2seq_logger_config
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
        self.sample_dep: List[float] = []
        self.token_dep = pa.array([])
        self.token_gold_probs = pa.array([])
        self.labels = labels

    @property
    def token_map_key(self) -> str:
        if self.split == Split.inference and self.inference_name is not None:
            return self.inference_name
        return str(self.split)

    def validate_and_format(self) -> None:
        """Validate the lengths, calculate token level dep, extract GT probs"""
        if self.labels is not None:
            self.labels = self._convert_tensor_ndarray(self.labels)
        self.logits = self._convert_tensor_ndarray(self.logits)
        self.ids = self._convert_tensor_ndarray(self.ids)
        assert len(self.ids) == len(self.logits), (
            "Must pass in a valid batch with equal id and logit length, got "
            f"id: {len(self.ids)},logits: {len(self.logits)}"
        )
        if self.labels is not None:
            assert len(self.labels) == len(self.ids), "TODO: Must be same len message"

        # Ground truth probs, including the padding for ignored labels
        # TODO: This is incredibly slow. This is what needs to be optimized. Can we
        #  potentially do this on the GPU with torch? And dont convert to a np array
        probs = self.convert_logits_to_probs(self.logits)
        (
            self.token_dep,
            self.sample_dep,
            self.token_gold_probs,
        ) = self.get_token_dep_probs(self.ids, probs)

    def get_dep_for_sample(
        self, sample_id: int, sample_probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts DEP per token prediction for a single sample

        Args:
            sample_id: The sample id
            sample_probs: The probabilities for each token in the sample
                sample_probs.shape is [max_token_len, vocab_size]

        Returns:
            dep: The DEP per token prediction for the sample
                dep.shape is [num_tokens_in_label]
            gold_probs: The probabilities of the GT token label for the sample
                gold_probs.shape is [num_tokens_in_label]
        """
        assert (
            self.logger_config.tokenizer is not None
        ), "Must set your tokenizer. Use `dq.set_tokenizer`"
        labels = self.logger_config.id_to_tokens[self.token_map_key][sample_id]
        if self.logger_config.tokenizer.padding_side == "left":
            probs = sample_probs[-len(labels) :]
        else:
            probs = sample_probs[: len(labels)]
        gold_probs = probs[np.arange(len(labels)), labels]
        probs_copy = probs.copy()
        probs_copy[np.arange(len(labels)), labels] = 0
        # Max non-gold probability
        max_probs = np.max(probs_copy, axis=-1)
        margin = gold_probs - max_probs
        dep = (1 - margin) / 2
        return dep, gold_probs

    def get_token_dep_probs(
        self, batch_ids: np.ndarray, batch_probs: np.ndarray
    ) -> Tuple[pa.array, List[float], pa.array]:
        """Extracts DEP per token prediction

        First, extract the probabilities of the GT token label

        Probs is a numpy array of shape [batch_size, max_token_len, vocab_size] where
        for each sample (text input) in the batch, every token of that sample has a
        probability vector of size vocab_size (which can be 30k+).

        We use advanced indexing to extract out only the probabilities for the token
        label for each sample, for each batch.

        Then, we get the second highest probabilities per token via similar indexing.

        Finally, compute dep and return.

        token_dep, token_probs, and labels are of shape
        [batch_size, max_token_length], but for each sample in the batch, the tokens
        for that sample that are ignored/padded are indexed out by this function.
        So we use that to get only the ones we care about.

        We return a pyarrow array because each batch will have a different shape, which
        can't be represented in numpy

        Returns: (batch_token_dep, batch_dep, batch_gold_probs)
            batch_token_dep: The DEP per token prediction for the batch
                len(batch_token_dep) == batch_size
                batch_token_dep[i].shape is [num_tokens_in_label]
            batch_dep: The DEP per sample for the batch
                len(batch_dep) == batch_size
            batch_gold_probs: The probabilities of the GT token label for the batch
                len(batch_gold_probs) == batch_size
                batch_gold_probs[i].shape is [num_tokens_in_label]
        """
        batch_token_deps = []
        batch_deps = []
        batch_gold_probs = []
        for sample_id, sample_probs in zip(batch_ids, batch_probs):
            token_dep, gold_probs = self.get_dep_for_sample(sample_id, sample_probs)
            batch_token_deps.append(token_dep)
            batch_deps.append(float(np.max(token_dep)))
            batch_gold_probs.append(gold_probs)
        return pa.array(batch_token_deps), batch_deps, pa.array(batch_gold_probs)

    def _get_data_dict(self) -> Dict:
        """Returns the data dictionary for writing to disk"""
        batch_size = len(self.ids)
        data = {
            C.id.value: self.ids,
            C.token_deps.value: self.token_dep,
            C.dep.value: self.sample_dep,
            C.token_gold_probs.value: self.token_gold_probs,
            C.split_.value: [Split[self.split].value] * batch_size,
            C.epoch.value: [self.epoch] * batch_size,
        }
        if self.split == Split.inference:
            data[C.inference_name.value] = [self.inference_name] * batch_size
        return data

    def _write_dict_to_disk(self, path: str, object_name: str, data: Dict) -> None:
        save_arrow_file(path, object_name, data)
