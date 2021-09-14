import inspect
from enum import Enum, unique
from typing import Any, List, Optional, Union

from dataquality.schemas.split import Split


@unique
class GalileoModelConfigAttributes(str, Enum):
    emb = "emb"
    probs = "probs"
    ids = "ids"
    # we need to ignore this because "split" is a builtin function in python
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
        assert (
            self.emb is not None and self.probs is not None and self.ids is not None
        ), (
            f"All of emb, probs, and ids for your GalileoModelConfig must be set, but "
            f"got emb:{bool(self.emb)}, probs:{bool(self.probs)}, ids:{bool(self.ids)}"
        )

        assert len(self.emb) == len(self.probs) == len(self.ids), (
            f"All of emb, probs, and ids for your GalileoModelConfig must be the same "
            f"length, but got (emb, probs, ids) "
            f"({len(self.emb)},{len(self.probs)}, {self.ids})"
        )
        if self.split:
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
        labels: List[Union[int, str]] = None,
        ids: List[Union[int, str]] = None,
    ) -> None:
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.text = text if text is not None else []
        self.labels = labels if labels is not None else []
        self.ids = ids if ids is not None else []

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
        * Text and Labels must both exist
        * Text and Labels must be the same length
        * If ids exist, it must be the same length as text/labels
        :return:
        """
        assert self.labels is not None and self.text is not None, (
            f"Both text and labels for your GalileoDataConfig must be set, but got "
            f"text:{bool(self.text)}, labels:{bool(self.labels)}"
        )

        assert len(self.text) == len(self.labels), (
            f"labels and text must be the same length, but got"
            f"(labels, text) ({len(self.labels)},{len(self.text)})"
        )

        if self.ids:
            assert len(self.ids) == len(self.labels), (
                f"Ids exists but are not the same length as text and labels. "
                f"(ids, text) ({len(self.ids)}, {len(self.text)})"
            )

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
    for attr, member_class in inspect.getmembers(cls):
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
    for attr, member_class in inspect.getmembers(cls):
        if isinstance(member_class, GalileoModelConfig):
            return attr
    raise AttributeError("No GalileoModelConfig attribute found!")
