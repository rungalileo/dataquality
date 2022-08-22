from pkg_resources import parse_version  # type: ignore

try:
    from tensorflow import __version__

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def is_tf_2() -> bool:
    if TF_AVAILABLE:
        return parse_version(__version__) == 2
    return False
