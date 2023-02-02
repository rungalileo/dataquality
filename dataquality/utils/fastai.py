from typing import Any, Dict
import pandas as pd
from fastai.data.load import DataLoader
from torch.nn import Module


def convert_dl_to_df(dl: DataLoader, x_col: str = "image") -> pd.DataFrame:
    """
    Converts a fastai DataLoader to a pandas DataFrame.
    :param dl: Fast ai DataLoader to convert.
    :param x_col: Name of the column to use for the x values, for example text or image.
    :return: Pandas DataFrame with the data from the DataLoader.
    """
    additonal_data = {}
    if x_col == "image":
        additonal_data["path"] = dl.items
    x = []
    y = []
    for x_item, y_item in dl.dataset:
        x.append(x_item)
        y.append(int(y_item))
    return pd.DataFrame({"id": range(dl.n), x_col: x, "label": y, **additonal_data})


def forward_hook_with_store(
    store: Dict[str, Any], layer: Module, model_input: Any, model_output: Any
) -> None:
    """
    Forward hook to store the output of a layer.
    :param store: Dictionary to store the output in.
    :param layer: Layer to store the output of.
    :param model_input: Input to the model.
    :param model_output: Output of the model.
    :return: None
    """
    store["model_input"] = model_input
    store["model_output"] = model_output
