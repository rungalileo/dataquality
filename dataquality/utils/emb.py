import os
import warnings
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import vaex
from huggingface_hub.utils import HfHubHTTPError
from pydantic import UUID4
from vaex.dataframe import DataFrame

from dataquality.clients.objectstore import ObjectStore
from dataquality.utils.file import (
    get_largest_epoch_for_splits,
    get_last_epoch_for_splits,
)
from dataquality.utils.hdf5_store import HDF5_STORE
from dataquality.utils.vaex import (
    add_umap_pca_to_df,
    create_data_embs_df,
    get_output_df,
)

object_store = ObjectStore()
DATA_EMB_PATH = "data_emb/data_emb.hdf5"


def get_concat_emb_df(run_dir: str, split_epoch: Dict[str, int]) -> DataFrame:
    """Returns the concatenated embedding df for all available (non inf) splits"""
    split_dfs = []
    for split_name, epoch in split_epoch.items():
        split_loc = f"{run_dir}/{split_name}"
        split_dfs.append(
            get_output_df(
                f"{split_loc}/{epoch}",
                prob_only=False,
                split=split_name,
                epoch_or_inf=epoch,
            )
        )
    return vaex.concat(split_dfs)


def save_processed_emb_dfs(
    df_emb: DataFrame, split_epoch: Dict[str, int], run_dir: str
) -> None:
    """Saves the concatenated embeddings to their respective split locations

    The df has a column that has the split name, so we filter the dataframe on split,
    and write it back to where we read it from, but now containing the pca and xy
    embeddings
    """
    for split_name, epoch in split_epoch.items():
        # We need to first save the df to a temporary location, because the dataframe
        # Was read (mmapped) from the original location, so it would fail if you tried
        # to write back to it
        split_loc = f"{run_dir}/{split_name}/{epoch}/{HDF5_STORE}"
        tmp_loc = f"{run_dir}/{split_name}/{epoch}/tmp_{HDF5_STORE}"
        df = df_emb[df_emb["split"] == split_name]
        df.export(tmp_loc)
        os.remove(split_loc)
        os.rename(tmp_loc, split_loc)


def apply_umap_to_embs(run_dir: str, last_epoch: Optional[int]) -> None:
    """In the event that the user has Nvidia cuml installed, we apply UMAP locally

    We take advantage of the user's GPU to apply UMAP significantly faster than if
    we were to do it on the Galileo server

    We want to only apply UMAP (and PCA) to the final epoch of the split (or the
    `last_epoch` if specified).
    """
    # Get the correct epoch to process for each split
    split_epoch = get_largest_epoch_for_splits(run_dir, last_epoch)
    # In the case of inference only
    if not split_epoch:
        return
    concat_df = get_concat_emb_df(run_dir, split_epoch)
    df_emb = add_umap_pca_to_df(concat_df)
    save_processed_emb_dfs(df_emb, split_epoch, run_dir)


def upload_umap_data_embs(
    project_id: UUID4,
    run_id: UUID4,
    input_data_dir: str,
    run_dir: str,
    last_epoch: Optional[int],
    data_embs_col: str,
) -> None:
    """Given the location to _all_ input text, create and upload the data embs

    Read in all of the text samples from all splits, create the data embeddings,
    apply PCA over all samples, then UMAP, then upload each split to its respective
    location in object storage.

    Data embeddings are _always_ stored in the final epoch of the dataset, so we
    use `get_last_epoch_for_splits` to find that for each split. Similarly, for
    inference, we store it in the inference name. So when the split is inference,
    we further filter the dataframe by inference name and upload the df.
    """
    df = vaex.open(f"{input_data_dir}/**/data*.arrow")
    try:
        df_emb = create_data_embs_df(df, text_col=data_embs_col, lazy=False)
    except HfHubHTTPError as e:
        warnings.warn(
            "Unable to download transformer from huggingface. Data embeddings "
            f"will be skipped. {str(e)}"
        )
        return
    split_epoch = get_last_epoch_for_splits(run_dir, last_epoch)
    df_emb = add_umap_pca_to_df(df_emb, data_embs=True)
    data_emb_cols = ["id", "emb", "emb_pca", "data_x", "data_y"]
    for split in df_emb["split"].unique():
        proj_run_split = f"{project_id}/{run_id}/{split}"
        df_split = df_emb[df_emb["split"] == split]
        # Upload for each split
        if split in ["training", "test", "validation"]:
            minio_file = f"{proj_run_split}/{split_epoch[split]}/{DATA_EMB_PATH}"
            df_split = df_split[data_emb_cols]
            object_store.create_project_run_object_from_df(df_split, minio_file)
        else:
            # We need to split for each inference name and upload individually
            for inf_name in df_split["inference_name"].unique():
                df_inf = df_split[df_split["inference_name"] == inf_name][data_emb_cols]
                minio_file = f"{proj_run_split}/{inf_name}/{DATA_EMB_PATH}"
                object_store.create_project_run_object_from_df(df_inf, minio_file)


def np_to_pa(embs: np.ndarray) -> pa.ListArray:
    """Converts a numpy column to pyarrow array"""
    if len(embs.shape) <= 2:
        # Faster than the below method, but only works for 1 and 2 dim arrays
        return pa.array(list(embs))
    else:
        return pa.ListArray.from_pandas(embs.tolist())


def convert_np_to_pa(df: DataFrame, col: str) -> DataFrame:
    """Vaex safe conversion of `np_to_pa` above

    This function allows us to safely convert a high dimensional numpy array to a
    pyarrow array. Since HDF5 files cannot store multi-dimensional numpy arrays,
    in certain cases, such as embeddings in Seq2Seq, we must store them
    as pyarrow arrays.

    We convert the column in a memory safe way using a vaex register function.
    """

    @vaex.register_function()
    def _np_to_pa(embs: np.ndarray) -> pa.ListArray:
        return np_to_pa(embs)

    df[col] = df[col]._np_to_pa()
    return df


def convert_pa_to_np(df: DataFrame, col: str) -> DataFrame:
    """Converts a pyarrow array col to numpy column

    This function allows us to safely convert a high dimensional pyarrow array to a
    numpy array. It is the inverse of `convert_np_to_pa` and is used when calculating
    things such as `similar_to` on the PCA embeddings, which assumes numpy arrays.

    We convert the column in a memory safe way using a vaex register function.

    While this fn is built primarily for 2d arrays, it will work for any dimensional
    array.
    """

    @vaex.register_function()
    def _pa_to_np(embs: pa.ChunkedArray) -> np.ndarray:
        # np.array(embs) leads to a numpy array of shape (dim1,)
        # np.array(embs.tolist()) leads to a numpy array of shape (dim1, dim2)
        return np.array(embs.to_pylist())

    df[col] = df[col]._pa_to_np()
    return df
