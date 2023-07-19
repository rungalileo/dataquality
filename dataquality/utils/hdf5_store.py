import os
import sys
from typing import Dict, List, Optional, Tuple, Type

import h5py
import numpy as np
import vaex
from vaex.arrow.convert import arrow_string_array_from_buffers as convert_bytes

from dataquality.utils import tqdm
from dataquality.utils.thread_pool import lock

HDF5_STORE = "hdf5_store.hdf5"


class HDF5Store(object):
    """
    Simple class to append value to a hdf5 file on disc.

    Used to concatenate HDF5 files for vaex

    Params:
        datapath: filepath of h5 file
        dataset: dataset name within the file
        shape: dataset shape (not counting main/batch axis)
        dtype: numpy dtype

    Usage:
        hdf5_store = HDF5Store('/tmp/hdf5_store.h5','X', shape=(20,20,3))
        x = np.random.random(hdf5_store.shape)
        hdf5_store.append(x)
        hdf5_store.append(x)

    Adapted From https://gist.github.com/wassname/a0a75f133831eed1113d052c67cf8633
    """

    def __init__(
        self,
        datapath: str,
        dataset: str,
        shape: Tuple,
        dtype: Type = np.float32,
        compression: Optional[str] = None,
        chunk_len: int = 1,
    ):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode="a") as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape,
            )

    def append(self, values: np.ndarray) -> None:
        with h5py.File(self.datapath, mode="a") as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + values.shape[0],) + self.shape)
            dset[self.i :] = values
            self.i += values.shape[0]
            h5f.flush()


def _save_hdf5_file(location: str, file_name: str, data: Dict) -> None:
    """
    Helper function to save a dictionary as an hdf5 file that can be read by vaex
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


def _valid_prob_col(col: str) -> bool:
    return (
        col.endswith("id")
        or "gold" in col
        or "pred" in col
        or "prob" in col  # encapsulates prob, conf_prob, and loss_prob
        or col.startswith("span")
        or col.startswith("galileo")
    )


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
        cols = [c for c in df.get_column_names() if _valid_prob_col(c)]
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

    for file in tqdm(
        files,
        leave=False,
        desc="Processing data for upload",
        file=sys.stdout,
    ):
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
