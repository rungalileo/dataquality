def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def hf_available() -> bool:
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        return False


def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401

        return True
    except ImportError:
        return False
