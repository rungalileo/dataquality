import sys

from dataquality.analytics import Analytics
from dataquality.exceptions import AmplitudeException

a = Analytics()


def test_log_galileo_exception(set_test_config, cleanup_after_use):
    raise AmplitudeException("test")


def test_log_general_exception(set_test_config, cleanup_after_use):
    pass


def test_import_keys(set_test_config, cleanup_after_use):
    modules = set(sys.modules)
    assert "dataquality" not in modules
    import dataquality as dq

    modules = set(sys.modules)
    assert "dataquality" in modules
    dq.__version__
