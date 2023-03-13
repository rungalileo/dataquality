import os
import shutil
import time
import warnings
from typing import Optional

from dataquality import GalileoWarning


def get_file_extension(path: str) -> str:
    """Returns the file extension"""
    return os.path.splitext(path)[-1]


def get_largest_epoch_for_split(split_dir: str, last_epoch: Optional[int]) -> int:
    """Gets the latest epoch that is largest in size

    In the event of early stopping, the last split won't be the same size as the others.
    This checks that, and returns the last epoch if no early stopping occured, otherwise
    the second to last epoch
    """
    if last_epoch is None:
        last_epoch = max([int(i) for i in os.listdir(split_dir)])
    if last_epoch == 0:
        return last_epoch
    last_epoch_size = os.path.getsize(f"{split_dir}/{last_epoch}")
    next_last_epoch_size = os.path.getsize(f"{split_dir}/{last_epoch-1}")
    return last_epoch if last_epoch_size >= next_last_epoch_size else last_epoch - 1


def _shutil_rmtree_retry(dir_path: str) -> None:
    """_shutil_rmtree_retry

    Attempts to remove a directory and all its contents.

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
