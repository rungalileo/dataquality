import inspect
from typing import Any, Dict, List, Optional, Tuple, Union


class GModelConfig:
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
    ) -> None:
        # Need to compare to None because they may be np arrays which cannot be
        # evaluated with bool directly
        self.emb = emb if emb is not None else []
        self.probs = probs if probs is not None else []
        self.ids = ids if ids is not None else []

    @staticmethod
    def get_valid() -> List[str]:
        """
        Returns a list of valid attributes that GModelConfig accepts
        :return: List[str]
        """
        return ["emb", "probs", "ids"]

    def dict(self) -> Dict[str, Any]:
        return dict(emb=self.emb, probs=self.probs, ids=self.ids)

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * emb, probs, and ids must exist and be the same length
        :return:
        """
        assert (
            self.emb is not None and self.probs is not None and self.ids is not None
        ), (
            f"All of emb, probs, and ids for your GModelConfig must be set, but got "
            f"emb:{bool(self.emb)}, probs:{bool(self.probs)}, ids:{bool(self.ids)}"
        )

        assert len(self.emb) == len(self.probs) == len(self.ids), (
            f"All of emb, probs, and ids for your GModelConfig must be the same "
            f"length, but got (emb, probs, ids) "
            f"({len(self.emb)},{len(self.probs)}, {self.ids})"
        )

    def is_valid(self) -> bool:
        """
        A function that returns if your GModelConfig is valid or not
        :return: bool
        """
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in GModelConfig.get_valid():
            raise AttributeError(
                f"{key} is not a valid attribute of GModelConfig. "
                f"Only {GModelConfig.get_valid()}"
            )
        super().__setattr__(key, value)


class GDataConfig:
    """
    Class for storing training data metadata to be logged to Galileo. Separate
    GDataConfigs will be created for training, validation, and testing data
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
    def get_valid() -> List[str]:
        """
        Returns a list of valid attributes that GModelConfig accepts
        :return: List[str]
        """
        return ["text", "labels", "ids"]

    def dict(self) -> Dict[str, Any]:
        return dict(text=self.text, labels=self.labels, ids=self.ids)

    def validate(self) -> None:
        """
        Validates that the current config is correct.
        * Text and Labels must both exist
        * Text and Labels must be the same length
        * If ids exist, it must be the same length as text/labels
        :return:
        """
        assert self.labels is not None and self.text is not None, (
            f"Both text and labels for your GDataConfig must be set, but got "
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
        A function that returns if your GDataConfig is valid or not
        :return: bool
        """
        try:
            self.validate()
            return True
        except AssertionError:
            return False

    def __setattr__(self, key: Any, value: Any) -> None:
        if key not in GDataConfig.get_valid():
            raise AttributeError(
                f"{key} is not a valid attribute of GDataConfig. "
                f"Only {GDataConfig.get_valid()}"
            )
        super().__setattr__(key, value)


def get_dataconfig_attr(cls: object) -> str:
    """
    Returns the attribute of a class that corresponds to the GDataConfig class.
    This assumes only 1 GDataConfig object exists in the class

    :param cls: The class
    :return: The attribute name
    """
    for attr, member_class in inspect.getmembers(cls):
        if isinstance(member_class, GDataConfig):
            return attr
    raise AttributeError("No GDataConfig attribute found!")


def get_modelconfig_attr(cls: object) -> str:
    """
    Returns the attribute of a class that corresponds to the GModelConfig class.
    This assumes only 1 GModelConfig object exists in the class

    :param cls: The class
    :return: The attribute name
    """
    for attr, member_class in inspect.getmembers(cls):
        if isinstance(member_class, GModelConfig):
            return attr
    raise AttributeError("No GModelConfig attribute found!")
