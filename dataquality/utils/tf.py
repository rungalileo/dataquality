from packaging import version

from dataquality.utils.imports import tf_available


def is_tf_2() -> bool:
    if tf_available():
        from tensorflow import __version__

        return version.parse(__version__).major == 2  # type: ignore
    return False
