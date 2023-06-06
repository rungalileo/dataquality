import os
import shutil
import time
import warnings
from typing import Dict, Optional

from dataquality.exceptions import GalileoWarning
from dataquality.schemas.split import Split


def get_file_extension(path: str) -> str:
    """Returns the file extension"""
    return os.path.splitext(path)[-1]


def _get_dir_size(dir_: str) -> int:
    """Returns dir size in bytes"""
    return sum(
        os.path.getsize(f"{dir_}/{f}") for f in os.listdir(dir_) if f.endswith(".hdf5")
    )


def get_largest_epoch_for_split(split_dir: str, last_epoch: Optional[int]) -> int:
    """Gets the latest epoch that is largest in size

    In the event of early stopping, the last split won't be the same size as the others.
    This checks that, and returns the last epoch if no early stopping occured, otherwise
    the second to last epoch
    """
    if last_epoch is None:
        last_epoch = max([int(i) for i in os.listdir(split_dir)])
    if not os.path.exists(f"{split_dir}/{last_epoch-1}"):
        return last_epoch
    last_epoch_size = _get_dir_size(f"{split_dir}/{last_epoch}")
    next_last_epoch_size = _get_dir_size(f"{split_dir}/{last_epoch-1}")
    return last_epoch if last_epoch_size >= next_last_epoch_size else last_epoch - 1


def get_largest_epoch_for_splits(
    run_dir: str, last_epoch: Optional[int]
) -> Dict[str, int]:
    """For each available (non inf) split, return the largest epoch in terms of bytes

    The 'largest' epoch is the last epoch in the split, unless early stopping occurred,
    in which case it's the second to last epoch

    :param run_dir: The location on disk to the run data
    :param last_epoch: The user can optionally tell us to only upload up to a specific
        epoch. If they did, this will indicate that
    """
    split_epoch = {}
    for split in [Split.train, Split.test, Split.validation]:
        split_loc = f"{run_dir}/{split}"
        if not os.path.exists(split_loc):
            continue
        split_epoch[split.value] = get_largest_epoch_for_split(split_loc, last_epoch)
    return split_epoch


def get_last_epoch_for_splits(
    run_dir: str, last_epoch: Optional[int]
) -> Dict[str, int]:
    """For each available (non inf) split, return the last epoch of training

    If `last_epoch` is provided, consider that as the last (if it exists)
    """
    split_epoch = {}
    for split in [Split.train, Split.test, Split.validation]:
        split_loc = f"{run_dir}/{split}"
        if not os.path.exists(split_loc):
            continue
        final_epoch = max([int(i) for i in os.listdir(split_loc)])
        epoch = min(last_epoch, final_epoch) if last_epoch is not None else final_epoch
        split_epoch[split.value] = epoch
    return split_epoch


def _shutil_rmtree_retry(dir_path: str) -> None:
    """_shutil_rmtree_retry

    Attempts to remove a directory and all its contents.

    This is because in certain environments (like nfs), the file takes time and retries
    to be deleted

    Args:
        dir_path (str): the path to the directory to remove
    """
    max_retry = 3
    retry = 0
    while retry < max_retry:
        try:
            shutil.rmtree(dir_path)
            return
        except OSError as e:
            warnings.warn(
                f"Failed to remove path:{dir_path} due to error:{e}. Trying again.",
                GalileoWarning,
            )
            retry += 1
            time.sleep(1)

    # If the directory still hasn't cleared, raise a warning
    if retry == max_retry:
        warnings.warn(f"Failed to remove path:{dir_path} after {max_retry} attempts.")
