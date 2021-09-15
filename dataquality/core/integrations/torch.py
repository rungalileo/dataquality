import warnings
from typing import Any, Union

import gorilla
import numpy as np
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import dataquality
from dataquality import GalileoException, config
from dataquality.core.integrations.config import (
    GalileoModelConfig,
    get_dataconfig_attr,
    get_modelconfig_attr,
)
from dataquality.schemas.split import Split

_GORILLA_WATCH_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=True)
_GORILLA_UNWATCH_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=False)


def watch(model: Module) -> None:
    """
    Function to instantiate autologging to Galileo into vanilla Pytorch models. Because
    Pytorch doesn't support hooks or callbacks, we manually patch the apply function
    of the users Pytorch model, enabling us to force the logging to Galileo.

    NOTE: This will turn on autologging for ALL models of this class

    In order for this function to work, users must utilize the GalileoModelConfig in
    their Pytorch model class `__init__ or `forward` function, and implement a
    `forward` function
    (see `dataquality.core.integrations.config.GalileoModelConfig`)

    :param model: The torch.nn.Module to watch
    :return: None
    """
    assert (
        config.current_project_id
    ), "You must initialize dataquality before invoking a callback!"
    assert (
        config.current_run_id
    ), "You must initialize dataquality before invoking a callback!"
    if not isinstance(model, Module):
        raise GalileoException(
            "Expected a pytorch model (torch.nn.Module). Received " + str(type(model))
        )

    if not hasattr(model, "forward"):
        raise GalileoException(
            "Your model must implement a forward function in order "
            "to enable automated logging to Galileo!"
        )

    def patch_forward(cls: Module, *args: Any, **kwargs: Any) -> Any:
        """
        A patched forward function
        :param cls: The model class
        """
        # Run the forward for the model before logging
        orig = gorilla.get_original_attribute(cls, "forward")
        res = orig(*args, **kwargs)
        try:
            config_attr = get_modelconfig_attr(cls)
        except AttributeError:  # User didn't specify a model config
            raise GalileoException(
                "Your model must utilize the GalileoModelConfig in"
                " order to enable automated logging to Galileo!"
            )
        model_config: GalileoModelConfig = getattr(cls, config_attr)

        try:
            model_config.validate()
        except AssertionError as e:
            warnings.warn(
                f"The provided GalileoModelConfig is invalid. Logging to "
                f"Galileo will be skipped. Config Error: {str(e)}"
            )
            return res

        if model_config.epoch is None:  # Compare to None because 0 will be False
            warnings.warn(
                "epoch must be set in your GalileoModelConfig for pytorch models to "
                "enable autologging to Galileo. If you are using Pytorch Lightning, "
                "consider using the DataQualityCallback in your trainer instead. "
                "Logging will be skipped."
            )
            return res

        split = model_config.split
        if not model_config.split:
            if not model.training:
                warnings.warn(
                    "either split must be set in your GalileoModelConfig or your model "
                    "must be in 'training' mode (calling `model.train()`) for pytorch "
                    "models to enable autologging to Galileo. If you are using, "
                    "Lightning consider using the DataQualityCallback in your trainer "
                    "instead."
                )
                return res
            else:
                warnings.warn(
                    "Model config split was not set, but training mode is "
                    "set in your model. Using split=training for logging to"
                    "Galileo"
                )
                split = "training"

        for id, prob, emb in zip(
            model_config.ids, model_config.probs, model_config.emb
        ):
            #
            # 🔭 Logging outputs with Galileo!
            #
            dataquality.log_model_output(
                {
                    "id": id,
                    "epoch": model_config.epoch,
                    "split": split,
                    "emb": emb,
                    "prob": prob,
                    "pred": str(int(np.argmax(prob))),
                }
            )
        return res

    patch = gorilla.Patch(
        model.__class__, "forward", patch_forward, settings=_GORILLA_WATCH_SETTINGS
    )
    gorilla.apply(patch)


def unwatch(model: Module) -> None:
    """
    Unwatch a model that has previously been watched. This will turn off autologging
    for all models of this class

    :param model: The model to unwatch
    :return: None
    """
    if not hasattr(model, "forward"):
        return
    original = gorilla.get_original_attribute(model, "forward")
    gorilla.Patch(
        model.__class__, "forward", original, settings=_GORILLA_UNWATCH_SETTINGS
    )


def log_input_data(data: Union[DataLoader, Dataset], split: str) -> None:
    """
    Log input data to Galileo

    :param data: DataSet or DataLoader for training/validation/testing
    :param split: The data split. One of (training, validation, test, inference)
    :return: None
    """
    if split not in Split.get_valid_attributes():
        raise GalileoException(
            f"split must be one of {Split.get_valid_attributes()} but got {split}"
        )

    try:
        if isinstance(data, Dataset):
            dataset_config = get_dataconfig_attr(data)
            data_config = getattr(data, dataset_config)
        elif isinstance(data, DataLoader):
            dataset_config = get_dataconfig_attr(data.dataset)
            data_config = getattr(data.dataset, dataset_config)
        else:
            raise GalileoException(
                f"data must be one of (Dataset, DataLoader) but got {type(data)}"
            )
    except AttributeError:
        raise GalileoException(
            "Could not find a GalileoDataConfig as a part of your "
            "dataset. You must include one to call this function."
        )
    try:
        data_config.validate(split=split)
    except AssertionError as e:
        raise GalileoException(
            f"The provided GalileoDataConfig is invalid. Logging to "
            f"Galileo cannot be performed. Config Error: {str(e)}"
        ) from None
    ids = data_config.ids if data_config.ids else range(len(data_config.text))
    for idx, text, label in zip(ids, data_config.text, data_config.labels):
        dataquality.log_input_data(
            {
                "id": idx,
                "text": text,
                "gold": str(label) if split != Split.inference else None,
                "split": split,
            }
        )
