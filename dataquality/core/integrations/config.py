import warnings
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Union

import numpy as np

from dataquality.schemas.jsonl_logger import TORCH_AVAILABLE
from dataquality.schemas.split import Split

if TORCH_AVAILABLE:
    from torch import Tensor

MAX_META_COLS = 50  # Limit the number of metadata attrs a user can log


@unique
class GalileoModelConfigAttributes(str, Enum):
    emb = "emb"
    probs = "probs"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    epoch = "epoch"

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoModelConfigAttributes))


@unique
class GalileoDataConfigAttributes(str, Enum):
    text = "text"
    labels = "labels"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataConfigAttributes))


class GalileoModelConfig:
    """
    Class for storing model metadata to be logged to Galileo.
    * Embeddings: List[List[Union[int,float]]]
    * Probabilities from forward passes during model training/evaluation.
    List[List[float]]
    * ids: Indexes of each input field: List[Union[int,str]]
    """

    def __init__(
        self,
        emb: List[List[Union[int, float]]] = None,
        probs: List[List[float]] = None,
        ids: List[Union[int, str]] = None,
        split: Optional[str] = None,
        epoch: Optional[int] = None,
    ) -> None:
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.epoch = epoch

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoModelConfigAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        emb_len = len(self.emb)
        prob_len = len(self.probs)
        id_len = len(self.ids)

        # We add validation here instead of requiring the params at init because
        # for lightning callbacks, we add these automatically for the user, so they
        # can create the config in their training loop and we will manage this metadata
        assert self.split, "Your GalileoModelConfig has no split!"
        assert self.epoch is not None, "Your GalileoModelConfig has no epoch!"

        self.emb = _convert_tensor_ndarray(self.emb, "Embedding")
        self.probs = _convert_tensor_ndarray(self.probs, "Prob")
        self.ids = _convert_tensor_ndarray(self.ids)

        assert emb_len and prob_len and id_len, (
            f"All of emb, probs, and ids for your GalileoModelConfig must be set, but "
            f"got emb:{bool(emb_len)}, probs:{bool(prob_len)}, ids:{bool(id_len)}"
        )

        assert emb_len == prob_len == id_len, (
            f"All of emb, probs, and ids for your GalileoModelConfig must be the same "
            f"length, but got (emb, probs, ids) -> ({emb_len},{prob_len}, {id_len})"
        )
        if self.split:
            # User may manually pass in 'train' instead of 'training'
            # but we want it to conform
            self.split = Split.training.value if self.split == "train" else self.split
            assert (
                isinstance(self.split, str)
                and self.split in Split.get_valid_attributes()
            ), (
                f"Split should be one of {Split.get_valid_attributes()} "
                f"but got {self.split}"
            )

        if self.epoch:
            assert isinstance(self.epoch, int), (
                f"If set, epoch must be int but was " f"{type(self.epoch)}"
            )

    def is_valid(self) -> bool:
        """
        A function that returns if your GalileoModelConfig is valid or not
        :return: bool
        """
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in GalileoModelConfig.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of GalileoModelConfig. "
                f"Only {GalileoModelConfig.get_valid_attributes()}"
            )
        super().__setattr__(key, value)


class GalileoDataConfig:
    """
    Class for storing training data metadata to be logged to Galileo. Separate
    GalileoDataConfigs will be created for training, validation, and testing data
    * text: The raw text inputs for model training. List[str]
    * labels: the ground truth labels aligned to each text field. List[Union[str,int]]
    * ids: Optional unique indexes for each record. If not provided, will default to
    the index of the record. Optional[List[Union[int,str]]]
    """

    def __init__(
        self,
        text: List[str] = None,
        labels: List[str] = None,
        ids: List[Union[int, str]] = None,
        split: str = None,
        **kwargs: Dict[str, List[Union[str, float, int]]],
    ) -> None:
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.labels = [str(i) for i in labels] if labels is not None else []
        self.ids = ids if ids is not None else []
        self.split = split
        self.meta = kwargs

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataConfigAttributes.get_valid()

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * Text and Labels must both exist (unless split is 'inference' in which case
        labels must be None)
        * Text and Labels must be the same length
        * If ids exist, it must be the same length as text/labels
        :return: None
        """

        label_len = len(self.labels)
        text_len = len(self.text)
        id_len = len(self.ids)

        self.text = _convert_tensor_ndarray(self.text)
        self.labels = _convert_tensor_ndarray(self.labels)
        self.ids = _convert_tensor_ndarray(self.ids)

        assert self.split, "Your GalileoDataConfig has no split!"
        self.split = Split.training.value if self.split == "train" else self.split
        assert (
            isinstance(self.split, str) and self.split in Split.get_valid_attributes()
        ), (
            f"Split should be one of {Split.get_valid_attributes()} "
            f"but got {self.split}"
        )
        if self.split == Split.inference:
            assert not len(
                self.labels
            ), "You cannot have labels in your inference split!"
        else:
            assert label_len and text_len, (
                f"Both text and labels for your GalileoDataConfig must be set, but got"
                f" text:{bool(text_len)}, labels:{bool(text_len)}"
            )

            assert text_len == label_len, (
                f"labels and text must be the same length, but got"
                f"(labels, text) ({label_len},{text_len})"
            )

        if self.ids:
            assert id_len == text_len, (
                f"Ids exists but are not the same length as text and labels. "
                f"(ids, text) ({id_len}, {text_len})"
            )
        else:
            self.ids = list(range(text_len))

        if len(self.meta.keys()) > MAX_META_COLS:
            warnings.warn(
                f"You can only log up to {MAX_META_COLS} metadata attrs. "
                f"The first {MAX_META_COLS} will be logged only."
            )
        # When logging metadata columns, if the user breaks a rule, don't fail
        # completely, just warn them and remove that metadata column
        # Cast to list for in-place dictionary mutation
        for key, values in list(self.meta.items())[:MAX_META_COLS]:
            # Key must not override a default
            if key in GalileoDataConfig.get_valid_attributes():
                warnings.warn(
                    f"Metadata column names must not override default values "
                    f"{GalileoDataConfig.get_valid_attributes()}. This metadata field "
                    f"will be removed."
                )
                self.meta.pop(key)
                continue
            # Must be the same length as input
            if len(values) != text_len:
                warnings.warn(
                    f"Expected {text_len} values for key {key} but got "
                    f"{len(values)}. Will not log this metadata column."
                )
                self.meta.pop(key)
                continue
            # Values must be a point, not an iterable
            valid_types = (str, int, float, np.floating, np.integer)
            invalid_values = filter(lambda t: not isinstance(t, valid_types), values)
            bad_val = next(invalid_values, None)
            if bad_val:
                warnings.warn(
                    f"Metadata column {key} has one or more invalid values {bad_val} "
                    f"of type {type(bad_val)}. Only strings and numbers can be logged."
                )
                self.meta.pop(key)
            # TODO - string values must be < MAX_STR_LEN characters

    def is_valid(self) -> bool:
        """
        A function that returns if your GalileoDataConfig is valid or not
        :return: bool
        """
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in GalileoDataConfig.get_valid_attributes():
            raise AttributeError(
                f"{key} is not a valid attribute of GalileoDataConfig. "
                f"Only {GalileoDataConfig.get_valid_attributes()}"
            )
        super().__setattr__(key, value)


def get_dataconfig_attr(cls: object) -> str:
    """
    Returns the attribute of a class that corresponds to the GalileoDataConfig class.
    This assumes only 1 GalileoDataConfig object exists in the class

    :param cls: The class
    :return: The attribute name
    """
    for attr in dir(cls):
        member_class = getattr(cls, attr)
        if isinstance(member_class, GalileoDataConfig):
            return attr
    raise AttributeError("No GalileoDataConfig attribute found!")


def get_modelconfig_attr(cls: object) -> str:
    """
    Returns the attribute of a class that corresponds to the GalileoModelConfig class.
    This assumes only 1 GalileoModelConfig object exists in the class

    :param cls: The class
    :return: The attribute name
    """
    for attr in dir(cls):
        member_class = getattr(cls, attr)
        if isinstance(member_class, GalileoModelConfig):
            return attr
    raise AttributeError("No GalileoModelConfig attribute found!")


def _convert_tensor_ndarray(arr: List, attr: Optional[str] = None) -> List:
    """Handles numpy arrays and tensors conversions"""
    if TORCH_AVAILABLE:
        if isinstance(arr, Tensor):
            arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        if attr in ("Embedding", "Prob"):
            shp = arr.shape
            assert len(shp) == 2, f"{attr} tensor must be 2D shape, but got shape {shp}"
    return list(arr)
