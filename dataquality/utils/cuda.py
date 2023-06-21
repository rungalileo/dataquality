from typing import Tuple

import numpy as np

from dataquality.clients.objectstore import ObjectStore

object_store = ObjectStore()

PCA_CHUNK_SIZE = 100_000
PCA_N_COMPONENTS = 100


def cuml_available() -> bool:
    try:
        import cuml  # noqa F401

        return True
    except Exception:
        return False


def get_pca_embeddings(embs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses Cuda and GPUs to create the PCA embeddings before uploading

    Should only be called if cuda ML available (`cuda_available()`)

    Returns the PCA embeddings, the components_ of the pca model, and the mean_ of
    the pca model
    """
    import cuml

    n_components = min(PCA_N_COMPONENTS, *embs.shape)
    pca = cuml.IncrementalPCA(n_components=n_components, batch_size=PCA_CHUNK_SIZE)
    emb_pca = pca.fit_transform(embs)
    return emb_pca, pca.components_, pca.mean_


def get_umap_embeddings(embs: np.ndarray) -> np.ndarray:
    """Uses Cuda and GPUs to create the UMAP embeddings before uploading

    Should only be called if cuda ML available (`cuda_available()`)
    """
    import cuml

    umap = cuml.UMAP(n_neighbors=15, n_components=2, min_dist=0.25)
    return umap.fit_transform(embs)
