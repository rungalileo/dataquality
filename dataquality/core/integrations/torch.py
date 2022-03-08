import warnings
from typing import Any, Union

import gorilla
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger import BaseGalileoModelLogger
from dataquality.schemas.split import Split

_GORILLA_WATCH_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=True)
_GORILLA_UNWATCH_SETTINGS = gorilla.Settings(allow_hit=True, store_hit=False)


def watch(model: Module) -> None:
    """
    Function to instantiate autologging to Galileo into vanilla Pytorch models. Because
    Pytorch doesn't support hooks or callbacks, we manually patch the apply function
    of the users Pytorch model, enabling us to force the logging to Galileo.

    NOTE: This will turn on autologging for ALL models of this class

    In order for this function to work, users must utilize the GalileoModelLogger in
    their Pytorch model class `__init__ or `forward` function, and implement a
    `forward` function

    :param model: The torch.nn.Module to watch
    :return: None
    """
    assert (
        config.current_project_id and config.current_run_id
    ), "You must initialize dataquality before invoking a callback!"
    if not isinstance(model, Module):
        raise GalileoException(
            f"Expected a pytorch model (torch.nn.Module). Received {str(type(model))}"
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
            logger_attr = BaseGalileoModelLogger.get_model_logger_attr(cls)
        except AttributeError:  # User didn't specify a model logger
            warnings.warn(
                "Your model must utilize the GalileoModelLogger in order to enable "
                "automated logging to Galileo! Logging will be skipped."
            )
            return res
        model_logger: BaseGalileoModelLogger = getattr(cls, logger_attr)

        # Compare to None because 0 will be False
        if model_logger.epoch is None and model_logger.logger_config.cur_epoch is None:
            warnings.warn(
                "epoch must be set in your GalileoModelLogger for pytorch models to "
                "enable autologging to Galileo. If you are using Pytorch Lightning, "
                "consider using the DataQualityCallback in your trainer instead. "
                "You can also set the epoch using `dataquality.set_epoch`"
                "Logging will be skipped."
            )
            return res

        if not model_logger.split and not model_logger.logger_config.cur_split:
            if not model.training:
                warnings.warn(
                    "either split must be set in your GalileoModelLogger or your model "
                    "must be in 'training' mode (calling `model.train()`) for pytorch "
                    "models to enable autologging to Galileo. If you are using "
                    "Lightning consider using the DataQualityCallback in your trainer "
                    "You can also set the split using `dataquality.set_split`"
                    "instead."
                )
                return res
            else:
                warnings.warn(
                    "Model logger split was not set, but training mode is "
                    "set in your model. Using split=training for logging to "
                    "Galileo"
                )
                model_logger.split = "training"

        try:
            model_logger.log()
        except GalileoException as e:
            warnings.warn(
                f"Logging model outputs to Galileo could not be "
                f"completed. See exception: {str(e)}"
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
            logger_attr = BaseGalileoDataLogger.get_data_logger_attr(data)
            data_logger = getattr(data, logger_attr)
        elif isinstance(data, DataLoader):
            logger_attr = BaseGalileoDataLogger.get_data_logger_attr(data.dataset)
            data_logger = getattr(data.dataset, logger_attr)
        else:
            raise GalileoException(
                f"data must be one of (Dataset, DataLoader) but got {type(data)}"
            )
    except AttributeError:
        raise GalileoException(
            "Could not find a GalileoDataLogger as a part of your "
            "Dataset. You must include one to call this function."
        )
    data_logger.split = split
    data_logger.log()
