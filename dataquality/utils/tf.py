from tensorflow import __version__
from pkg_resources import parse_version


def is_tf_2() -> bool:
    return parse_version(__version__) == 2
