import os
from typing import Dict, List, Union

import numpy as np
import pyarrow as pa
import vaex
from sklearn.decomposition import IncrementalPCA
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException
from dataquality.loggers.base_logger import BaseLoggerAttributes
from dataquality.schemas.split import Split
from dataquality.utils.cuda import (
    PCA_CHUNK_SIZE,
    PCA_N_COMPONENTS,
    cuml_available,
    get_pca_embeddings,
    get_umap_embeddings,
)
from dataquality.utils.hdf5_store import HDF5_STORE, concat_hdf5_files
from dataquality.utils.helpers import galileo_verbose_logging

object_store = ObjectStore()

# To decide between "all-MiniLM-L6-v2" or "all-mpnet-base-v2"
# https://www.sbert.net/docs/pretrained_models.html#model-overview
GALILEO_DATA_EMBS_ENCODER = "GALILEO_DATA_EMBS_ENCODER"
DEFAULT_DATA_EMBS_MODEL = "all-MiniLM-L6-v2"

COMPONENTS = "components"
MEAN = "mean"
PCA_COMPONENTS_NAME = "components.hdf5"
PCA_COMPONENTS_OBJECT_PATH = f"pca/{PCA_COMPONENTS_NAME}"
PCA_MEAN_NAME = "mean.hdf5"
PCA_MEAN_OBJECT_PATH = f"pca/{PCA_MEAN_NAME}"


def _join_in_out_frames(
    in_df: DataFrame, out_df: DataFrame, allow_missing_in_df_ids: bool = False
) -> DataFrame:
    """Helper function to join our input and output frames"""
    in_frame = in_df.copy()
    # There is an odd vaex bug where sometimes we lose the continuity of this dataframe
    # it's hard to reproduce, only shows up on linux, and hasn't been pinpointed yet
    # but materializing the join-key column fixes the issue
    # https://github.com/vaexio/vaex/issues/1972
    in_frame["id"] = in_frame["id"].values
    out_frame = out_df.copy()
    in_out = out_frame.join(in_frame, on="id", how="inner", lsuffix="_L").copy()
    if len(in_out) != len(out_frame) and not allow_missing_in_df_ids:
        num_missing = len(out_frame) - len(in_out)
        missing_ids = set(out_frame["id"].unique()) - set(in_out["id_L"].unique())
        split = out_frame["split"].unique()[0]
        raise GalileoException(
            "It seems there were logged outputs with no corresponding inputs logged "
            f"for split {split}. {num_missing} corresponding input IDs are missing:\n"
            f"{missing_ids}"
        )
    elif allow_missing_in_df_ids:
        # If we're downsampling, we make sure the out and in have an id intersection
        # and then we drop the out rows that don't have a corresponding in
        in_ids = set(in_frame["id"].unique())
        out_ids = set(out_frame["id"].unique())
        id_intersection = np.array(list(in_ids.intersection(out_ids)))
        in_out = in_out[in_out["id_L"].isin(id_intersection)]

    keep_cols = [c for c in in_out.get_column_names() if not c.endswith("_L")]
    in_out = in_out[keep_cols]
    return in_out


def validate_unique_ids(df: DataFrame, epoch_or_inf_name: str) -> None:
    """Helper function to validate the logged df has unique ids

    Fail gracefully otherwise
    """
    if df["id"].nunique() != len(df):
        epoch_or_inf_value, split = df[[epoch_or_inf_name, "split"]][0]
        dups = get_dup_ids(df)
        exc = (
            f"It seems your logged output data has duplicate ids in split {split}. If "
            f"you've re-run a block of code or notebook cell that logs model outputs, "
            f"that could be the cause. It could also be a misconfiguration in your "
            f"model architecture. Try reinitializing with `dq.init` to clear your "
            f"local environment, and then logging your data again. "
        )
        if galileo_verbose_logging():
            exc += (
                f"split:{split}, {epoch_or_inf_name}: {epoch_or_inf_value}, "
                f"dup ids and counts: {dups}"
            )
        raise GalileoException(exc)


def _upload_pca_data(components: np.ndarray, mean: np.ndarray) -> None:
    project_id, run_id = config.current_project_id, config.current_run_id
    # Save the components as an hdf5 file
    components_file = f"/tmp/{PCA_COMPONENTS_NAME}"
    components_object_path = f"{project_id}/{run_id}/{PCA_COMPONENTS_OBJECT_PATH}"
    vaex.from_dict({COMPONENTS: components}).export(components_file)
    bucket = config.results_bucket_name
    object_store.create_object(
        components_object_path, components_file, progress=False, bucket_name=bucket
    )

    # Save the mean as an hdf5 file
    mean_file_path = f"/tmp/{PCA_MEAN_NAME}"
    mean_object_path = f"{project_id}/{run_id}/{PCA_MEAN_OBJECT_PATH}"
    vaex.from_dict({MEAN: mean}).export(mean_file_path)
    object_store.create_object(
        mean_object_path, mean_file_path, progress=False, bucket_name=bucket
    )


def get_dup_ids(df: DataFrame) -> List:
    """Gets the list of duplicate IDs in a dataframe, if any"""
    df_copy = df.copy()
    dup_df = df_copy.groupby(by="id", agg="count")
    return dup_df[dup_df["count"] > 1].to_records()


def drop_empty_columns(df: DataFrame) -> DataFrame:
    """Drops any columns that have no values"""
    if len(df) == 0:
        return df
    df_copy = df.copy()
    cols = df.get_column_names()
    # Don't need to check the default columns, they've already been validated
    cols = [c for c in cols if c not in list(BaseLoggerAttributes)]
    col_counts = df.count(cols)
    empty_cols = [col for col, col_count in zip(cols, col_counts) if col_count == 0]
    for c in empty_cols:
        df_copy = df_copy.drop(c)
    return df_copy


def filter_df(df: DataFrame, col_name: str, value: str) -> DataFrame:
    """Filter vaex df on the value of a column

    Drop any columns for this df that are empty
    (e.g. metadata logged for a different split)
    """
    df_slice = df[df[col_name].str.equals(value)].copy()
    df_slice = drop_empty_columns(df_slice)
    # Remove the mask, work with only the filtered rows
    return df_slice.extract()


def rename_df(df: DataFrame, columns: Dict) -> DataFrame:
    """Renames a vaex df using a mapping"""
    df_copy = df.copy()
    for old, new in columns.items():
        df_copy.rename(old, new)
    return df_copy


def add_umap_pca_to_df(df: DataFrame, data_embs: bool = False) -> DataFrame:
    """Adds the PCA embeddings and UMAP xy embeddings if possible

    If data_embs is True, the x and y values from umap will be named data_x and data_y
    """
    if not cuml_available():
        return df
    dfc = df.copy()
    note = "[data embs]" if data_embs else "[embs]"
    print(f"{note} Found cuda ML libraries")
    print(f"{note} Applying dimensionality reduction step 1/2")
    emb_pca, components, mean = get_pca_embeddings(dfc["emb"].to_numpy())
    print(f"{note} Applying dimensionality reduction step 2/2")
    emb_xy = get_umap_embeddings(emb_pca)
    x, y = ("data_x", "data_y") if data_embs else ("x", "y")
    dfc["emb_pca"] = emb_pca
    # We save the components and mean of the PCA model to minio
    # see utils/emb.py::apply_umap_to_embs
    if not data_embs:
        _upload_pca_data(components, mean)
    dfc[x] = emb_xy[:, 0]
    dfc[y] = emb_xy[:, 1]
    return dfc


def create_data_embs_df(df: DataFrame, lazy: bool = True) -> DataFrame:
    """Runs sentence transformer on raw text to get off the shelf data embeddings

    :param df: The dataframe to get data embeddings for. Must have text col
    :param lazy: If true, we lazily apply the model to encode the text
    """
    # This import takes up to 25 seconds, so we don't want to eagerly import it
    import transformers
    from sentence_transformers import SentenceTransformer

    transformers.logging.disable_progress_bar()
    sentence_encoder = os.getenv(GALILEO_DATA_EMBS_ENCODER, DEFAULT_DATA_EMBS_MODEL)
    data_model = SentenceTransformer(sentence_encoder)
    transformers.logging.enable_progress_bar()
    df_copy = df.copy()

    @vaex.register_function()
    def apply_sentence_transformer(text: pa.array) -> np.ndarray:
        return data_model.encode(text.to_pylist(), show_progress_bar=False).astype(
            np.float32
        )

    if lazy:
        df_copy["emb"] = df_copy["text"].apply_sentence_transformer()
        df_copy = df_copy[["id", "emb"]]
    else:
        import torch

        # Downcasts to float16 where possible, speeds up processing by 10 it/sec
        with torch.autocast("cuda"):
            df_copy["emb"] = data_model.encode(
                df_copy["text"].tolist(), show_progress_bar=True
            ).astype(np.float32)

    return df_copy


def get_output_df(
    dir_name: str,
    prob_only: bool,
    split: str,
    epoch_or_inf: Union[str, int],
) -> DataFrame:
    """Creates the single hdf5 file for the output data of a split/epoch

    Applies the necessary conversions post-concatenation of files
    (see `concat_hdf5_files`)
    """
    out_frame_path = f"{dir_name}/{HDF5_STORE}"
    # It's possible the files were already concatenated and handled. In that case
    # just open the processed file
    if os.path.isfile(out_frame_path):
        return vaex.open(out_frame_path)
    str_cols = concat_hdf5_files(dir_name, prob_only)
    out_frame = vaex.open(out_frame_path)

    if split == Split.inference:
        dtype: Union[str, None] = "str"
        epoch_or_inf_name = "inference_name"
    else:
        dtype = None
        epoch_or_inf_name = "epoch"

    # Post concat, string columns come back as bytes and need conversion
    for col in str_cols:
        out_frame[col] = out_frame[col].as_arrow().astype("str")
        out_frame[col] = out_frame[f'astype({col}, "large_string")']
    if prob_only:
        out_frame["split"] = vaex.vconstant(split, length=len(out_frame), dtype="str")
        out_frame[epoch_or_inf_name] = vaex.vconstant(
            epoch_or_inf, length=len(out_frame), dtype=dtype
        )
    return out_frame


def add_pca_to_df(df: DataFrame, chunk_size: int = PCA_CHUNK_SIZE) -> DataFrame:
    """Adds the 'emb_pca' to the dataframe"""
    df_copy = df.copy()
    # n_components must be <= num samples and <= emb dimension
    n_components = min(PCA_N_COMPONENTS, *df["emb"].shape)
    pca = IncrementalPCA(n_components=n_components)
    for i1, i2, chunk in df_copy.evaluate_iterator("emb", chunk_size=chunk_size):
        pca.partial_fit(chunk)

    @vaex.register_function()
    def apply_pca(emb: np.ndarray) -> np.ndarray:
        return pca.transform(emb).astype(np.float32)

    df_copy["emb_pca"] = df_copy["emb"].apply_pca()
    # Fully apply the PCA model now, without ever bringing the full embeddings into
    # memory, only the PCA embeddings (much smaller)
    df_copy = df_copy.materialize("emb_pca")
    return df_copy
