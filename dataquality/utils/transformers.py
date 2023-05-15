import inspect
from typing import Any, Callable, Dict, List, Union

from torch.nn import Module
from transformers import Trainer

from dataquality.utils.patcher import Patch


class RemoveIdCollatePatch(Patch):
    _original_collate_fn: Union[Callable, Any]
    name = "remove_id_collate_patch"

    def __init__(
        self,
        trainer_cls: Trainer,
        keep_cols: List[str],
        store: Dict[str, List[int]],
        fn_name: str = "data_collator",
    ):
        """Initializes the class with a collate function,
        columns to keep, and a store to save ids.

        :param collate_fn: The collate function to wrap
        :param keep_cols: The columns to keep
        :param store: The store to use to save the ids
        """
        self.orig_cls = trainer_cls
        self._original_collate_fn = getattr(trainer_cls, fn_name)
        self._fn_name = fn_name
        self.keep_cols = keep_cols
        self.store = store
        self.patch()

    def _patch(self) -> "Patch":
        """Wraps the original collate function with the id removal functionality."""
        setattr(self.orig_cls, self._fn_name, self)
        return self

    def _unpatch(self) -> None:
        """Restores the original collate function,
        removing the id removal functionality."""
        self.orig_cls.collate_fn = self._original_collate_fn

    def __call__(self, rows: List[dict]) -> List[dict]:
        """Removes the id from each row and pass the cleaned version on.
        :param rows: The rows to clean
        :return: The cleaned rows
        """
        ids = []
        clean_rows = []
        for row in rows:
            clean_row = {}
            for key, value in row.items():
                if key == "id":
                    ids.append(value)
                elif key in self.keep_cols:
                    clean_row[key] = value
                elif len(self.keep_cols) == 0:
                    clean_row[key] = value
            clean_rows.append(clean_row)
        self.store["ids"] = ids
        return self._original_collate_fn(clean_rows)


class SignatureColumnsPatch(Patch):
    name = "signature_columns_patch"

    def __init__(self, trainer: Trainer):
        """Patches the trainer to add the id column to the signature columns.
        And adds a collate function to remove the id column. During training.
        If the id is already in the signature column, it will not be removed
            on unpatching.
        :param trainer: The trainer to patch"""
        self.id_in_signature = False
        self.trainer = trainer
        self.patch()

    def _patch(self) -> "Patch":
        """Patches the trainer to add the id column to the signature columns.
        And stores them in self.new_signature_columns."""
        self.new_signature_columns = self._add_id_to_signature_columns(self.trainer)
        return self

    def _add_id_to_signature_columns(self, trainer: Trainer) -> List[str]:
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
            else:
                self.id_in_signature = True

            return trainer._signature_columns
        return []

    def _unpatch(self) -> None:
        """Restores the original collate function"""
        if (
            isinstance(self.trainer._signature_columns, list)
            and not self.id_in_signature
        ):
            self.trainer._signature_columns.remove("id")


class ModelHook(Patch):
    __name__ = "dq_hook"
    name = "dq_hook"

    def __init__(self, model: Module, hook_fn: Callable) -> None:
        self.model = model
        self.hook_fn = hook_fn

    def _patch(self) -> "Patch":
        self.hook = self.model.register_forward_hook(self)
        return self

    def __call__(self, module: Module, input: Any, output: Any) -> None:
        self.hook_fn(module, input, output)

    def _unpatch(self) -> None:
        self.hook.remove()
