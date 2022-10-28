import re
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from dataquality.exceptions import GalileoException
from dataquality.utils.helpers import hook


def remove_id_collate_fn_wrapper(collate_fn: Any, store: Any) -> Callable:
    """Removes the id from each row and pass the cleaned version on.
    Will be used as a wrapper for the collate function.
    :param collate_fn: The collate function to wrap
    :param store: The store to use to save the ids (currently passed by reference)
    :return: The wrapped collate function
    """

    def remove_id(rows: List) -> List:
        """Removes the id from each row and pass the cleaned
        version on."""
        data, idx = rows[0]
        assert isinstance(idx, int), GalileoException(
            "id is missing in dataset. Wrap your dataset"
            "with dq.wrap_dataset(train_dataset)"
        )
        indices = []
        cleaned_rows = []
        for row in rows:
            indices.append(row[1])
            cleaned_rows.append(row[0])
        store["ids"] = indices
        return collate_fn(cleaned_rows)

    return remove_id


# store indices
def store_batch_indices(store: Dict) -> Callable:
    def process_batch_indices(
        next_index_func: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        """Stores the indices of the batch"""
        indices = next_index_func(*args, **kwargs)
        if indices:
            store["ids"] = indices
        return indices

    return process_batch_indices


# add patch to the dataloader iterator
def patch_iterator_with_store(store: Dict) -> Callable:
    """Patches the iterator of the dataloader to return the indices"""

    def patch_iterator(
        orig_iterator: Callable, *args: Tuple, **kwargs: Dict[str, Any]
    ) -> Callable:
        iteraror = orig_iterator(*args, **kwargs)
        iteraror._next_index = hook(iteraror._next_index, store_batch_indices(store))
        return iteraror

    return patch_iterator


def validate_fancy_index_str(input_str: str = "[:, 1:, :]") -> bool:
    """Validates a fancy index string.
    :param input_str: The string to validate for example "[:, 1:, :]"
    :return: True if the string is valid, False otherwise
    """
    valid = re.fullmatch(r"[\s,\[\]\d:\(\)]+", input_str)
    if valid:
        return True
    return False


def convert_fancy_idx_str_to_slice(
    fstr: str = "[:, 1:, :]",
) -> Union[tuple, slice, int]:
    """Converts a fancy index string to a slice.
    :param fstr: The fancy index string to convert for example "[:, 1:, :]"
    :return: The slice for example:
    (slice(None, None, None), slice(1, None, None), slice(None, None, None))
    """
    clean_str = fstr
    # Remove outer brackets
    if fstr[0] == "[" and fstr[-1] == "]":
        clean_str = clean_str[1:-1]
    # Validate the string before we eval the fancy index
    assert validate_fancy_index_str(
        clean_str
    ), 'Fancy index string is not valid. Valid format: "[:, 1:, :]"'
    return eval("np.s_[{}]".format(clean_str))


class HookManager:
    """
    Manages hooks for models. Has the ability to find the layer automatically.
    Otherwise the layer or the layer name needs to be provided.
    """

    # Stores all hooks to remove them from the model later.
    hooks: List[RemovableHandle] = []

    def get_embedding_layer_auto(self, model: Module) -> Module:
        """
        Use a scoring algorithm to find the embedding layer automatically
        given a model. The higher the score the more likely it is the embedding layer.
        """
        name, layer = next(model.named_children())
        print(f"Selected layer for the last hidden state embedding {name}")
        return layer

    def get_embedding_layer_by_name(self, model: Module, name: str) -> Module:
        """
        Iterate over each layer and stop once the the layer name matches
        :param model: Model
        :parm name: string
        """
        queue: Queue = Queue()
        start = model.named_children()
        queue.put(start)
        layer_names = []
        layer_names_str: str = ""
        while not queue.empty():
            named_children = queue.get()
            for layer_name, layer_model in named_children:
                layer_names.append(layer_name)
                layer_names_str = ", ".join(layer_names)
                if layer_name == name:
                    print(
                        f"Found layer {layer_name} in model layers: {layer_names_str}"
                    )
                    return layer_model
                layer_model._get_name()
                queue.put(layer_model.named_children())
        raise GalileoException(
            f"Layer could not be found in {layer_names_str}, "
            "make sure to check capitalization"
        )

    def attach_embedding_hook(
        self,
        model: Module,
        model_layer: Optional[Union[Module, str]] = None,
        embedding_hook: Callable = print,
    ) -> RemovableHandle:
        """Attach hook and save it in our hook list"""
        if model_layer is None:
            selected_layer = self.get_embedding_layer_auto(model)
        elif isinstance(model_layer, str):
            selected_layer = self.get_embedding_layer_by_name(model, model_layer)
        else:
            selected_layer = model_layer
        return self.attach_hook(selected_layer, embedding_hook)

    def attach_hook(self, selected_layer: Module, hook: Callable) -> RemovableHandle:
        """Register a hook and save it in our hook list"""
        h = selected_layer.register_forward_hook(hook)
        self.hooks.append(h)
        return h

    def remove_hook(self) -> None:
        """Remove all hooks from the model"""
        for h in self.hooks:
            h.remove()
