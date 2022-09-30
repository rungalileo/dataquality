import inspect
from typing import Any, Callable, List

import pandas as pd
from datasets import Dataset
from transformers import Trainer

def remove_keys(row:dict,keep_cols:List[str],id_col:str)->dict:
    id = row.pop(id_col,None)
    # Remove all keys that are not in the keep_cols from the row
    if len(keep_cols) > 0:
        for key in list(row.keys()):
            if key not in keep_cols:
                row.pop(key)
    return id

def remove_id_collate_fn_wrapper(collate_fn: Any,signature_columns:List[str], store: Any) -> Callable:
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
        store["ids"] = [remove_keys(row,signature_columns,"id") for row in rows]
        return collate_fn(rows)
    return remove_id




def add_id_to_signature_columns(trainer: Trainer) -> None:
    """Adds the signature columns to the trainer.
    Usually the signature columns are label and label_ids.
    This function will add them if they are not already present.
    Additionally it will add the id column if it is not present.
    :param trainer: The trainer to add the id column to.
    """
    # Taken from the trainer source code
    if not trainer.args.remove_unused_columns:
        return []

    # Taken from the trainer source code
    if trainer._signature_columns is None:
        # Inspect model forward signature to keep only the arguments it accepts.
        signature = inspect.signature(trainer.model.forward)
        trainer._signature_columns = list(signature.parameters.keys())
        # Labels may be named label or label_ids, the default data collator handles that.
        trainer._signature_columns += list(set(["label", "label_ids"] + trainer.label_names))

    # Here we add the ids so they won't get removed
    if type(trainer._signature_columns) is list:
        if "id" not in trainer._signature_columns:  # type: ignore
            trainer._signature_columns.append("id")  # type: ignore
    return trainer._signature_columns
