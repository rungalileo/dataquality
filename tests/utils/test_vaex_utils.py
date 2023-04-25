import numpy as np
import vaex

from dataquality.utils.cuda import PCA_N_COMPONENTS
from dataquality.utils.vaex import add_pca_to_df, drop_empty_columns
from tests.assets.constants import NUMPY_SEED

np.random.seed(NUMPY_SEED)


def test_drop_empty_cols() -> None:
    df1 = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9])
    df1["split"] = vaex.vconstant("training", length=len(df1))
    df2 = vaex.from_arrays(x=[1, 2, 3], y=[4, 5, 6], xz=[7, 8, 9])
    df2["split"] = vaex.vconstant("test", length=len(df2))

    in_frame = vaex.concat([df2, df1])
    split_df = in_frame[in_frame["split"] == "test"]
    assert split_df["z"].ismissing().to_numpy().all()
    split_df = drop_empty_columns(split_df)
    assert "z" not in split_df.get_column_names()


def test_add_pca_to_df() -> None:
    # DETR returns embeddings of dimension 2048.
    df = vaex.from_arrays(
        x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9], emb=np.random.rand(3, 2048)
    )
    df = add_pca_to_df(df)

    pca_embs = df["emb_pca"].values
    assert pca_embs.shape == (3, PCA_N_COMPONENTS)
    assert np.any(pca_embs[:, :3])
    assert not np.any(pca_embs[:, 3:])
