import numpy as np


def cuml_available() -> bool:
    try:
        import cuml  # noqa F401

        return True
    except ImportError:
        return False


def get_pca_embeddings(embs: np.ndarray) -> np.ndarray:
    """Uses Cuda and GPUs to create the PCA embeddings before uploading

    Should only be called if cuda ML available (`cuda_available()`)
    """
    import cuml

    pca = cuml.PCA(n_components=100)
    return pca.fit_transform(embs)


def get_umap_embeddings(embs: np.ndarray) -> np.ndarray:
    """Uses Cuda and GPUs to create the UMAP embeddings before uploading

    Should only be called if cuda ML available (`cuda_available()`)
    """
    import cuml

    umap = cuml.UMAP(n_neighbors=15, n_components=2, min_dist=0.25)
    return umap.fit_transform(embs)
