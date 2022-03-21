import os
import threading
from typing import Dict, List

import h5py
import numpy as np
import vaex
from vaex.arrow.convert import arrow_string_array_from_buffers as convert_bytes
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseLoggerAttributes
from dataquality.utils import tqdm
from dataquality.utils.hdf5_store import HDF5_STORE, HDF5Store

lock = threading.Lock()


def _save_hdf5_file(location: str, file_name: str, data: Dict) -> None:
    """
    Helper function to save a dictionary as an hdf5 file
    """
    if not os.path.isdir(location):
        with lock:
            os.makedirs(location, exist_ok=True)
    file_path = f"{location}/{file_name}"
    with h5py.File(file_path, "w") as f:
        for col in data:
            group = f.create_group(f"/table/columns/{col}")
            col_data = np.array(data[col])
            if None in col_data:
                # h5py expects np.nan instead of None
                col_data = col_data.astype(np.float_)

            # String columns
            ctype = col_data.dtype
            if not np.issubdtype(ctype, np.number) and not np.issubdtype(
                ctype, np.bool_
            ):
                dtype = h5py.string_dtype()
                col_data = col_data.astype(dtype)
            else:
                dtype = col_data.dtype

            shape = col_data.shape
            group.create_dataset(
                "data", data=col_data, dtype=dtype, shape=shape, chunks=shape
            )


def _join_in_out_frames(in_df: DataFrame, out_df: DataFrame) -> DataFrame:
    """Helper function to join our input and output frames"""
    in_frame = in_df.copy()
    # There is an odd vaex bug where sometimes we lose the continuity of this dataframe
    # it's hard to reproduce, only shows up on linux, and hasn't been pinpointed yet
    # but materializing the join-key column fixes the issue
    # https://github.com/vaexio/vaex/issues/1972
    in_frame["id"] = in_frame["id"].values
    out_frame = out_df.copy()
    in_out = out_frame.join(
        in_frame, on="id", how="inner", lsuffix="_L", rsuffix="_R"
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
        dups = get_dup_ids(df)
        raise GalileoException(
            "It seems as though you do not have unique ids in this "
            f"split/epoch. Did you provide your own IDs? Or did you log for the same "
            f"Split multiple times? Check where you are logging {split} data\n"
            f"split:{split}, epoch:{epoch}, dup ids and counts:{dups}"
        )


def valid_ids(df: DataFrame) -> bool:
    """Returns whether or not a dataframe has unique IDs"""
    try:
        _validate_unique_ids(df)
        return True
    except GalileoException:
        return False


def get_dup_ids(df: DataFrame) -> List:
    """Gets the list of duplicate IDs in a dataframe, if any"""
    df_copy = df.copy()
    dup_df = df_copy.groupby(by="id", agg="count")
    return dup_df[dup_df["count"] > 1].to_records()


def concat_hdf5_files(location: str, prob_only: bool) -> List[str]:
    """Concatenates all hdf5 in a directory using an HDF5 store

    Vaex stores a dataframe as an hdf5 file in a predictable format using groups

    Each column gets its own group, following "/table/columns/{col}/data

    We can exploit that by concatenating our datasets with that structure, so vaex
    can open the final file as a single dataframe

    :param location: The directory containing the files
    :param prob_only: If True, only the id, prob, and gold columns will be concatted
    """
    str_cols = []
    stores = {}
    files = os.listdir(location)
    df = vaex.open(f"{location}/{files[0]}")

    # Construct a store per column
    if prob_only:
        cols = ["id"]
        cols += [c for c in df.get_column_names() if c.startswith("prob")]
        cols += [c for c in df.get_column_names() if c.startswith("gold")]
        cols += [c for c in df.get_column_names() if c.endswith("_gold")]
        cols += [c for c in df.get_column_names() if c.endswith("_pred")]
    else:
        cols = df.get_column_names()
    for col in cols:
        group = f"/table/columns/{col}/data"
        cval = df[col].to_numpy()
        if cval.ndim == 2:
            shape = cval[0].shape
        else:
            shape = ()
        dtype = df[col].dtype.numpy
        if not np.issubdtype(dtype, np.number) and not np.issubdtype(dtype, np.bool_):
            dtype = h5py.string_dtype(encoding="utf-8")
            str_cols.append(col)
        stores[col] = HDF5Store(f"{location}/{HDF5_STORE}", group, shape, dtype=dtype)

    print("Combining batches for upload")
    for file in tqdm(files):
        fname = f"{location}/{file}"
        with h5py.File(fname, "r") as f:
            dset = f["table"]["columns"]
            keys = dset.keys()
            keys = [key for key in keys if key in cols]
            for key in keys:
                col_data = dset[key]
                # We have a string column, need to parse it
                if "indices" in col_data.keys():
                    assert key in str_cols, f"Unexpected string column ({key}) found"
                    indcs = col_data["indices"][:]
                    data = col_data["data"][:]
                    d = convert_bytes(data, indcs, None).to_numpy(zero_copy_only=False)
                else:
                    d = col_data["data"][:]
                if key in str_cols:
                    d = d.astype(h5py.string_dtype(encoding="utf-8"))
                stores[key].append(d)
        os.remove(fname)
    return str_cols


def drop_empty_columns(df: DataFrame) -> DataFrame:
    """Drops any columns that have no values"""
    cols = df.get_column_names()
    # Don't need to check the default columns, they've already been validated
    cols = [c for c in cols if c not in list(BaseLoggerAttributes)]
    col_counts = df.count(cols)
    empty_cols = [col for col, col_count in zip(cols, col_counts) if col_count == 0]
    return df.drop(*empty_cols) if empty_cols else df
