from packaging import version

try:
    from tensorflow import __version__

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def is_tf_2() -> bool:
    if TF_AVAILABLE:
        return version.parse(__version__).major == 2  # type: ignore
    return False
