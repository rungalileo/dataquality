import inspect
from typing import Any, Callable, List

import pandas as pd
from datasets import Dataset
from transformers import Trainer


def pre_process_dataset(data: Dataset) -> pd.DataFrame:
    """Converts a dataset to a pandas dataframe

    :param data: Dataset (huggingface) to convert
    """
    # Load the labels in a dictionary
    labels = data.features["label"].names
    labels = {v: k for v, k in enumerate(labels)}

    # Load the train data into a frame
    data_df = pd.DataFrame.from_dict(data)
    data_df["label"] = data_df["label"].map(labels)
    if "id" not in data_df.columns:
        data_df["id"] = data_df.index
    return data_df


def remove_id_collate_fn_wrapper(collate_fn: Any, store: Any) -> Callable:
    """Removes the id from each row and pass the cleaned version on.
    Will be used as a wrapper for the collate function.
    :param collate_fn: The collate function to wrap
    :param store: The store to use to save the ids (currently passed by reference)
    :return: The wrapped collate function
    """

    def remove_id(rows: List[dict]) -> List[dict]:
        """Removes the id from each row and pass the cleaned version on.
        :param rows: The rows to clean
        :return: The cleaned rows
        """
        # Remove id by reference
        store["ids"] = [row.pop("id", None) for row in rows]
        return collate_fn(rows)

    return remove_id


def add_id_col_to_dataset(dataset: Dataset) -> Dataset:
    """Adds the id column to the dataset. Assumes it is equal to the index.
    :param dataset: The dataset to add the id column to
    :return: The dataset with the id column
    """
    return dataset.add_column("id", list(range(len(dataset))))


def add_id_to_signature_columns(trainer: Trainer) -> None:
    """Adds the signature columns to the trainer.
    Usually the signature columns are label and label_ids.
    This function will add them if they are not already present.
    Additionally it will add the id column if it is not present.
    :param trainer: The trainer to add the id column to.
    """
    # Taken from the trainer source code
    if trainer._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(trainer.model.forward)
        trainer._signature_columns = list(signature.parameters.keys())  # type: ignore
        # Labels may be named label or label_ids,
        # the default data collator handles that.
        trainer._signature_columns += ["label", "label_ids"]  # type: ignore
    # Here we add the ids so they won't get removed
    if type(trainer._signature_columns) is list:
        if "id" not in trainer._signature_columns:  # type: ignore
            trainer._signature_columns.append("id")  # type: ignore
