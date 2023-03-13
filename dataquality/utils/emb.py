import os
from typing import Dict, Optional

import vaex
from vaex.dataframe import DataFrame

from dataquality.utils.file import get_largest_epoch_for_splits
from dataquality.utils.hdf5_store import HDF5_STORE
from dataquality.utils.vaex import add_umap_pca_to_df, get_output_df


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
    concat_df = get_concat_emb_df(run_dir, split_epoch)
    df_emb = add_umap_pca_to_df(concat_df)
    save_processed_emb_dfs(df_emb, split_epoch, run_dir)
