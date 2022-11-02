import inspect
from typing import Any, Callable, Dict, List, Union

from transformers import Trainer


def remove_id_collate_fn_wrapper(
    collate_fn: Union[Callable, Any], keep_cols: List[str], store: Dict[str, List[int]]
) -> Callable:
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
        ids = []
        clean_rows = []
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if key == "id":
                    ids.append(value)
                # Only keep the columns that are not in the signature columns
                elif key in keep_cols:
                    clean_row[key] = value
                # If the signature columns are not set, keep all columns
                elif len(keep_cols) == 0:
                    clean_row[key] = value
            clean_rows.append(clean_row)
        store["ids"] = ids
        return collate_fn(clean_rows)

    return remove_id


def add_id_to_signature_columns(trainer: Trainer) -> List[str]:
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
        # Labels may be named label or label_ids,
        # the default data collator handles that.
        signature_keys = list(signature.parameters.keys()) + ["label", "label_ids"]
        signature_cols = list(set(signature_keys + trainer.label_names))
        trainer._signature_columns = signature_cols  # type: ignore

    # Here we add the ids so they won't get removed
    if isinstance(trainer._signature_columns, list):
        if "id" not in trainer._signature_columns:
            trainer._signature_columns.append("id")
        return trainer._signature_columns
    return []
