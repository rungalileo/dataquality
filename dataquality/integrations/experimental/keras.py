from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras

import dataquality as dq
from dataquality import config
from dataquality.analytics import Analytics
from dataquality.clients.api import ApiClient
from dataquality.exceptions import GalileoException
from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig
from dataquality.schemas.split import Split
from dataquality.utils.helpers import wrap_fn
from dataquality.utils.keras import (
    combine_with_default_kwargs,
    generate_indices,
    generate_split,
    get_x_len,
    patch_layer_call,
    save_input,
    save_output,
)

a = Analytics(ApiClient, config)
a.log_import("integrations/experimental/keras")


class DataQualityCallback(keras.callbacks.Callback):
    store: Dict[str, Any]
    model: tf.keras.layers.Layer
    logger_config: BaseLoggerConfig

    def __init__(
        self,
        store: Dict[str, Any],
        logger_config: BaseLoggerConfig,
        log_function: Callable,
        model: tf.keras.layers.Layer,
        *args: Tuple,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize the callback by passing in the model and the input store.
        :param store: The store to save the input and output to.
        :param model: The model to patch.
        """
        a.log_function("keras/dqcallback")
        self.log_function = log_function
        self.store = store
        self.logger_config = logger_config
        self.model = model
        super(DataQualityCallback, self).__init__()
        if not tf.executing_eagerly():
            raise GalileoException("Needs to be executing eagerly")

    def on_train_begin(self, logs: Any = None) -> None:
        """Initialize the training by extracting the model input arguments.
        and from it generate the indices of the batches."""
        assert self.model.run_eagerly, GalileoException(
            "Model must be compiled with run_eagerly=True"
        )
        self.fit_kwargs = combine_with_default_kwargs(
            self.model.fit,  # model fit function for default kwargs
            self.store["model_fit"].get("args", ()),  # model fit args
            self.store["model_fit"].get("kwargs"),  # model fit kwargs
        )
        x_len = get_x_len(self.fit_kwargs.get("x"))
        if x_len is None:
            self.x_len_is_none = True
        else:
            # If that data is splitted we perform the same splitting as keras
            self.fit_kwargs.update(generate_split(x_len, self.fit_kwargs))
            # Now we can generate the indices of the batches
            self.store["train_indices"] = generate_indices(
                x_len,
                batch_size=self.fit_kwargs["batch_size"],
                steps_per_epoch=self.fit_kwargs["steps_per_epoch"],
                epochs=self.fit_kwargs["epochs"],
                shuffle=self.fit_kwargs["shuffle"],
                sample_weight=self.fit_kwargs["sample_weight"],
                class_weight=self.fit_kwargs["class_weight"],
                initial_epoch=self.fit_kwargs["initial_epoch"],
            )

    def on_epoch_begin(self, epoch: int, logs: Dict) -> None:
        """At the beginning of the epoch we set the epoch in the store.
        :param epoch: The epoch number.
        :param logs: The logs."""
        dq.set_epoch(epoch)

    def _clear_logger_config_helper_data(self) -> None:
        """Clear the helper data from the logger config for the current step."""
        self.store.pop("input", None)
        self.store.pop("output", None)
        self.store.pop("indices_ids", None)

    def on_train_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        """At the beginning of the batch we clear the
        helper data from the logger config."""
        self._clear_logger_config_helper_data()
        dq.set_split(Split.train)

    def on_train_batch_end(self, batch: Any, logs: Dict = None) -> None:
        """At the end of the batch we log the input of the classifier and the output.
        :param batch: The batch number.
        :param logs: The logs."""
        dq.set_split(Split.train)
        # We try to fetch the indices directly from the store
        ids = self.store.get("indices_ids")
        # If we don't have them we try to fetch them from the generated indices
        if ids is None:
            epoch = self.logger_config.cur_epoch
            step = batch
            ids = self.store.get("train_indices", {}).get(f"epoch_{epoch}_{step}")
            assert ids is not None, GalileoException("No logged indices found")

        self.log_function(
            embs=self.store["input"],
            logits=self.store["output"],
            ids=ids,
        )

    def on_test_begin(self, logs: Any = None) -> None:
        """At the beginning of the test we set the split to test.
        And generate the indices of the batches."""
        dq.set_split(Split.validation)
        x_len = get_x_len(self.fit_kwargs.get("val_x"))
        if x_len is None:
            self.val_x_len_is_none = True
        else:
            self.store["validation_indices"] = generate_indices(
                x=x_len,
                batch_size=self.fit_kwargs.get("validation_batch_size")
                or self.fit_kwargs["batch_size"],
                steps_per_epoch=self.fit_kwargs.get("validation_steps"),
                sample_weight=self.fit_kwargs.get("val_sample_weight"),
            )

    def on_test_batch_begin(self, batch: Any, logs: Dict = None) -> None:
        """At the beginning of the batch we clear the helper
        data from the logger config."""
        self._clear_logger_config_helper_data()
        dq.set_split(Split.validation)

    def on_test_batch_end(self, batch: Any, logs: Dict = None) -> None:
        """At the end of the validation batch we log the input of the classifier
        and the output."""
        dq.set_split(Split.validation)
        ids = self.store.get("indices_ids")

        if ids is None:
            step = batch
            ids = self.store.get("validation_indices", {}).get(f"epoch_0_{step}")
            assert ids is not None, GalileoException("No logged indices found")

        self.log_function(
            embs=self.store["input"],
            logits=self.store["output"],
            ids=ids,
        )

    def on_predict_begin(self, batch: int) -> None:
        """At the beginning of the prediction we set the split to test."""
        dq.set_split(Split.test)
        dq.set_epoch(0)
        predict_kwargs = self.store["model_predict"].get("kwargs")
        predict_args = self.store["model_predict"].get("args", ())

        predict_kwargs = combine_with_default_kwargs(
            self.model.predict, predict_args, predict_kwargs
        )

        x_len = get_x_len(predict_kwargs.get("x"))
        if x_len is None:
            self.predict_x_len_is_none = True
        else:
            self.store["test_indices"] = generate_indices(
                x=x_len,
                batch_size=predict_kwargs.get("batch_size"),
                steps_per_epoch=predict_kwargs.get("steps"),
            )

    def on_predict_batch_end(self, batch: int, logs: Any = None) -> None:
        dq.set_split(Split.test)
        ids = self.store.get("indices_ids")
        if ids is None:
            step = batch
            ids = self.store.get("test_indices", {}).get(f"epoch_0_{step}")
            assert ids is not None, GalileoException("No loggeg indices found")

        self.log_function(
            embs=self.store["input"],
            logits=self.store["output"],
            ids=ids,
        )


def patch_model_fit_args_kwargs(store: Dict[str, Any], callback: Callable) -> Callable:
    """Store the args and kwargs of model.fit in the store.
    Adds the callback to the callbacks of the model.
    :param store: The store for the kwargs and args.
    :param callback: The callback to add to the model.
    :return: The patched model.fit function."""

    def fit_wrapper(orig_func: Callable, *args: Any, **kwargs: Any) -> None:
        """Stores the indices of the batch"""
        store["args"] = args
        store["kwargs"] = kwargs
        if kwargs.get("callbacks"):
            kwargs["callbacks"].append(callback)
        else:
            kwargs["callbacks"] = [callback]
        return orig_func(*args, **kwargs)

    return fit_wrapper


def store_model_ids(store: Dict[str, Any]) -> Callable:
    """Stores the indices of the batch. For a prebatched dataset"""

    def train_step_wrapper(orig_func: Callable, *args: Any, **kwargs: Any) -> None:
        """We pop out the ids from the batch dict and store them in the store."""
        try:
            ids = None
            if isinstance(args[0], tuple):
                ids = args[0]
                if not isinstance(ids, dict):
                    ids = ids[0]
                if isinstance(ids, dict):
                    ids = ids.pop("id", None)
                else:
                    ids = None
            store["indices_ids"] = ids
        except AttributeError:
            pass

        return orig_func(*args, **kwargs)

    return train_step_wrapper


def select_model_layer(
    model: tf.keras.layers.Layer, layer: Optional[Union[tf.keras.layers.Layer, str]]
) -> tf.keras.layers.Layer:
    """Selects the classifier layer from the model.
    :param model: The model.
    :param layer: The layer to select. If the layer with the name
    'classifier' is selected."""
    chosen_layer = layer
    if isinstance(chosen_layer, str) is None:
        for model_layer in model.layers:
            if model_layer.name == layer:
                chosen_layer = model_layer
                break
    else:
        for model_layer in model.layers:
            if model_layer.name == "classifier":
                chosen_layer = model_layer
                break
    assert chosen_layer is not None, GalileoException("Layer could not be found")
    assert not isinstance(chosen_layer, str), GalileoException(
        "Layer could not be found"
    )
    return chosen_layer


def _watch(
    logger_data: Dict[str, Any],
    logger_config: BaseLoggerConfig,
    log_function: Callable,
    model: tf.keras.layers.Layer,
    layer: Any = None,
    seed: int = 42,
) -> None:
    """Internal watch function that is used to watch the model during training.
    :param logger_data: The store for temporary logger data
    :param logger_config: The configuration of the logger
    :param log_function: The function that is used to log the data
    :param model: The model that is watched
    :param layer: The layer that is watched
    :param seed: The seed that is used for the random generator
    """
    # If we don't set the seed here, the random generator will be different for
    # each process and the indices will be different
    tf.random.set_seed(seed)
    # We add the callback to the model
    callback = DataQualityCallback(
        logger_data, logger_config, log_function, dq.log_model_outputs, model
    )

    # We store all monkey patches here so we can remove them later
    logger_data["model_patches"] = {
        "predict": model.predict,
        "test_step": model.test_step,
        "predict_step": model.predict_step,
        "__call__": model.__call__,
        "fit": model.fit,
    }
    # We store the args and kwargs of the model fit and prediction method in our store
    logger_data["model_fit"] = {}
    logger_data["model_predict"] = {}
    model.fit = wrap_fn(
        model.fit, patch_model_fit_args_kwargs(logger_data["model_fit"], callback)
    )
    model.predict = wrap_fn(
        model.predict,
        patch_model_fit_args_kwargs(logger_data["model_predict"], callback),
    )
    # We need to patch all functions to store the indices of the batch
    model.train_step = wrap_fn(model.train_step, store_model_ids(logger_data))
    model.test_step = wrap_fn(model.test_step, store_model_ids(logger_data))
    model.predict_step = wrap_fn(model.predict_step, store_model_ids(logger_data))
    model.__call__ = wrap_fn(model.__call__, store_model_ids(logger_data))
    # Select the layer that is watched
    chosen_layer = select_model_layer(model, layer)
    patch_layer_call(
        chosen_layer,
        before_call=partial(save_input, logger_data),
        after_call=partial(save_output, logger_data),
    )


def watch(model: tf.keras.layers.Layer, layer: Any = None, seed: int = 42) -> None:
    """Watch a model and log the inputs and outputs of a layer.
    :param model: The model to watch
    :param layer: The layer to watch, if None the classifier layer is used
    :param seed: The seed to use for the model"""
    if not getattr(model, "_dq", False):
        model._dq = True
    else:
        raise GalileoException(
            "Model is already being watched, run unwatch(model) first"
        )
    logger_data = dq.get_model_logger().logger_config.helper_data
    logger_config = dq.get_data_logger().logger_config
    _watch(logger_data, logger_config, dq.log_model_outputs, model, layer, seed)


def unwatch(model: tf.keras.layers.Layer) -> None:
    """Unpatches the model. Run after the run is finished
    :param model: The model to unpatch"""
    if not getattr(model, "_dq", False):
        raise GalileoException("Model is not watched, run watch(model) first")

    logger_data = dq.get_model_logger().logger_config.helper_data
    patched_layers = logger_data.get("model_patches", {})
    for k, v in patched_layers.items():
        print("unpatching", k)
        setattr(model, k, v)

    for layer in model.layers:
        if hasattr(layer, "_old_call"):
            layer.call = layer._old_call
            del layer._old_call
            del layer._before_call
            del layer._after_call
    if hasattr(model, "_dq"):
        del model._dq
