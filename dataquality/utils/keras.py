import inspect
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

import tensorflow as tf
from keras.engine import data_adapter
from tensorflow import keras

from dataquality.exceptions import GalileoException
from dataquality.utils.tf import is_tf_2

# If this is TF 1.x
if not is_tf_2():
    raise GalileoException(
        "Tensorflow/Keras 2.6 or above is required for this integration"
    )


def build_empty_model() -> tf.keras.Model:
    model = keras.Sequential([])
    model.compile(loss="mse", run_eagerly=True)
    return model


def generate_indices(
    x: Union[int, tf.Tensor],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Generates the indices for the training and validation data.
    :param x_len: The length of the training or validation data.
    :param kwargs: The arguments to the fit method. Kwargs needed are:
        - batch_size
        - epochs
        - steps_per_epoch
        - class_weight
        - sample_weight
        - initial_epoch
    :return: A dictionary of epochs/steps including the indices.
    """
    empty_model = build_empty_model()
    epochs = {}
    filtered_keys = ["x", "y", "workers", "use_multiprocessing", "model", "distribute"]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in filtered_keys}
    if isinstance(x, int):
        x = tf.range(x)
    data_handler = data_adapter.DataHandler(
        steps_per_execution=None,
        x=x,
        model=empty_model,
        use_multiprocessing=False,
        workers=1,
        distribute=False,
        **filtered_kwargs,
    )

    for epoch, iterator in data_handler.enumerate_epochs():
        for step in data_handler.steps():
            epochs[f"epoch_{epoch}_{step}"] = next(iterator)
    return epochs


def generate_split(
    x_len: int,
    kwargs: Dict[str, Any],
) -> Dict:
    validation_split = kwargs.get("validation_split")
    validation_data = kwargs.get("validation_data")
    sample_weight = kwargs.get("sample_weight")
    x = tf.range(x_len)
    y = x

    if validation_split and validation_data is None:
        # Create the validation data using the training data. Only supported
        # for `Tensor` and `NumPy` input.
        (x, y, sample_weight,), validation_data = data_adapter.train_validation_split(
            (x, y, sample_weight), validation_split=validation_split
        )

    if validation_data:
        (
            val_x,
            val_y,
            val_sample_weight,
        ) = data_adapter.unpack_x_y_sample_weight(validation_data)
        val_x = tf.range(len(val_x))
    else:
        val_x = None
        val_sample_weight = None
    return {
        "x": x,
        "y": None,
        "sample_weight": sample_weight,
        "val_x": val_x,
        "val_y": None,
        "val_sample_weight": val_sample_weight,
    }


def combine_with_default_kwargs(
    func: Callable, args: Tuple = (), kwargs: Dict[str, Any] = {}
) -> Dict[str, Any]:
    """Combines the default kwargs with the provided kwargs.
    While incoperating the args.
    :param signature: The signature of the function.
    :param args: The args to the function.
    :param kwargs: The kwargs to the function.
    :return: A dictionary of the combined kwargs.
    """
    # Combines args, kwargs with the default kwargs in three steps.
    # 1. Start with default values of the function.
    # 2. Update with the default values with the args.
    # 3. Add the kwargs if the key is in the signature.
    signature = inspect.signature(func)
    signature_keys = list(signature.parameters.keys())
    combined_kwargs = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    combined_kwargs.update({signature_keys[i]: arg for i, arg in enumerate(args)})
    combined_kwargs.update({k: v for k, v in kwargs.items() if k in signature_keys})
    return combined_kwargs


def get_x_len(x: Any) -> Optional[int]:
    try:
        return len(x)
    except TypeError:
        return None


# Monkey patching into the keras layer call method
# https://github.com/tensorflow/tensorflow/issues/33478#issuecomment-843720638


def patch_proxy_call(
    input: tf.Tensor, obj: tf.keras.layers.Layer, *args: Tuple, **kwargs: Any
) -> Any:
    """Call the patched layer method (before call and after call).
    :param input: The input to the layer.
    :param obj: The layer object.
    :param args: The arguments to the layer.
    :param kwargs: The keyword arguments to the layer.
    :return: The output of the layer.
    """

    # before call we can do something with the input
    # for example, we can save it to the store
    if callable(getattr(obj, "_before_call")):
        obj._before_call(obj, input)
    # call the original layer
    output = obj._old_call(input)
    # after call we can do something with the output
    if callable(getattr(obj, "_after_call")):
        hook_result = obj._after_call(obj, input, output)
        # if the hook returns something, we use it as the output
        if hook_result is not None:
            output = hook_result
    # return the output / patched output of the layer
    return output


def patch_layer_call(
    layer: tf.keras.layers.Layer,
    before_call: Optional[Callable] = None,
    after_call: Optional[Callable] = None,
) -> None:
    """Patch the layer call method to add before and after call hooks.
    :param layer: The layer to patch.
    :param before_call: The before call hook.
    :param after_call: The after call hook.
    """
    layer._before_call = before_call
    layer._after_call = after_call
    layer._old_call = layer.call
    layer.call = partial(patch_proxy_call, obj=layer)


def save_input(
    store: Dict[str, Any],
    layer: tf.keras.layers.Layer,
    input: tf.Tensor,
    *args: Tuple,
    **kwargs: Any,
) -> None:
    """Save the input to the store.
    :param store: The store to save the input to.
    :param layer: The layer that is being called.
    :param input: The input to the layer.
    :param args: The arguments to the layer.
    :param kwargs: The keyword arguments to the layer.
    """
    if input is not None:
        store["input"] = input


def save_output(
    store: Dict[str, Any],
    layer: tf.keras.layers.Layer,
    input: tf.Tensor,
    output: tf.Tensor,
) -> None:
    """Save the output to the store.
    :param store: The store to save the output to.
    :param layer: The layer that is being called.
    :param input: The input to the layer.
    :param output: The output of the layer.
    """
    if output is not None:
        store["output"] = output
