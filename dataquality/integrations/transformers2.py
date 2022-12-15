from typing import Dict, Any, Callable, Iterator, Iterable, List, Union
import numpy as np
from functools import wraps
from torch.utils.data.dataloader import _BaseDataLoaderIter, DataLoader
from torch.utils.data.sampler import Sampler, RandomSampler

import gc


def patch_dataloader(dataloader: DataLoader, store: Dict) -> None:
    def wrap_next_index(func: Callable, key: str = "ids") -> Callable:
        @wraps(func)
        def patched_next_index(*args: Any, **kwargs: Any) -> Any:
            indices = func(*args, **kwargs)
            if indices and key in store:
                store[key].append(indices)
            return indices

        return patched_next_index

    if hasattr(_BaseDataLoaderIter, "_patched"):
        # logger warning if already patched
        print("BaseDataLoaderIter already patched")
        return
    if "patches" not in store:
        store["patches"] = []

    store["patches"].append({"class": _BaseDataLoaderIter, "attr": "_next_index"})
    setattr(_BaseDataLoaderIter, "_old__next_index", _BaseDataLoaderIter._next_index)
    setattr(
        _BaseDataLoaderIter,
        "_next_index",
        wrap_next_index(_BaseDataLoaderIter._next_index, "ids"),
    )
    setattr(_BaseDataLoaderIter, "_patched", True)


def unpatch(store: Dict) -> None:
    # unpatch all instances and classes
    # starting with all classes
    for patch in store.get("patches", []):
        print("unpatching", patch["class"])
        if hasattr(patch["class"], "_patched"):
            print("unpatching", patch["class"], patch["attr"])
            setattr(
                patch["class"],
                patch["attr"],
                getattr(patch["class"], f"_old_{patch['attr']}"),
            )
            delattr(patch["class"], f"_old_{patch['attr']}")
            delattr(patch["class"], "_patched")
        # then all instances
        for obj in gc.get_objects():
            if isinstance(obj, patch["class"]) and hasattr(obj, "_patched"):
                print("unpatching", obj, patch["attr"])
                setattr(obj, patch["attr"], getattr(obj, f"old_{patch['attr']}"))
                delattr(obj, f"old_{patch['attr']}")
                delattr(obj, "_patched")


class BatchSamplerWithLogging(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    # batch_sampler = BatchSamplerWithLogging(
    #   RandomSampler(torch.arange(len(a))), 3, False, store)

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
        store: Dict,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.store = store

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    self.store["sampler_ids"].append(batch)
                    self.store["sampler_ids"].append(batch)
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    self.store["sampler_ids"].append(batch)
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                self.store["sampler_ids"].append(batch)
                yield batch[:idx_in_batch]
