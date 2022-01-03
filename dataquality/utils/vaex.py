import os
import threading
from collections import Counter
from glob import glob
from typing import List
from uuid import uuid4

import vaex
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.utils.thread_pool import ThreadPoolManager

lock = threading.Lock()


def _save_hdf5_file(location: str, file_name: str, file: DataFrame) -> None:
    """
    Helper function to save a vaex dataframe as an hdf5 file.
    """
    with lock:
        if not os.path.isdir(location):
            os.makedirs(location)
        file_path = f"{location}/{file_name}"
        file.export_hdf5(file_path)


def _try_concat_df(location: str) -> None:
    """Tries to concatenate dataframes eagerly during the logging process

    Multiple threads cannot concatenate dataframe simultaneously so we first check
    if anyone is concatenating (booleans are thread safe). If not, we concatenate.
    If yes, we simply pass.
    """
    if ThreadPoolManager.can_concat:
        # Only one thread can concat files at a time, but allow other threads to
        # continue writing new files
        ThreadPoolManager.can_concat = False
        with lock:  # Ensure we don't read while a thread is writing
            files = glob(f"{location}/*.hdf5")
        if len(files) > 25:
            new_name = f"{str(uuid4()).replace('-', '')[:12]}.hdf5"
            new_file = f"{location}/{new_name}"
            files_to_concat = _get_smallest_n_files(files, len(files) - 1)
            vaex.open_many(files_to_concat).export_hdf5(new_file)
            for file in files_to_concat:
                os.remove(file)
        ThreadPoolManager.can_concat = True


def _get_smallest_n_files(files: List[str], n: int) -> List[str]:
    file_sizes = []
    for f in files:
        size = os.stat(f).st_size
        file_sizes.append((f, size))
    file_sizes = sorted(file_sizes, key=lambda r: r[1])
    smallest_files = [r[0] for r in file_sizes[:n]]
    return smallest_files


def _join_in_out_frames(in_df: DataFrame, out_df: DataFrame) -> DataFrame:
    """Helper function to join our input and output frames"""
    in_frame = in_df.copy()
    out_frame = out_df.copy()
    in_frame["split_id"] = in_frame["split"] + in_frame["id"].astype("string")
    out_frame["split_id"] = out_frame["split"] + out_frame["id"].astype("string")
    in_out = out_frame.join(
        in_frame, on="split_id", how="inner", lsuffix="_L", rsuffix="_R"
    ).copy()
    if len(in_out) != len(out_frame):
        num_missing = len(out_frame) - len(in_out)
        missing_ids = set(out_frame["id"].unique()) - set(in_out["id_L"].unique())
        split = out_frame["split"].unique()[0]
        raise GalileoException(
            "It seems there were logged outputs with no corresponding inputs logged "
            f"for split {split}. {num_missing} corresponding input IDs are missing:\n"
            f"{missing_ids}"
        )
    keep_cols = [c for c in in_out.get_column_names() if not c.endswith("_L")]
    in_out = in_out[keep_cols]
    for c in in_out.get_column_names():
        if c.endswith("_R"):
            in_out.rename(c, c.rstrip("_R"))
    return in_out


def _validate_unique_ids(df: DataFrame) -> None:
    """Helper function to validate the logged df has unique ids

    Fail gracefully otherwise
    """
    if df["id"].nunique() != len(df):
        epoch, split = df[["epoch", "split"]][0]
        all_ids: List[int] = df["id"].tolist()
        dup_ids = [i for i, count in Counter(all_ids).items() if count > 1]
        raise GalileoException(
            "It seems as though you do not have unique ids in this "
            f"split/epoch. Did you provide your own IDs?\n"
            f"split:{split}, epoch:{epoch}, dup ids:{dup_ids}"
        )
