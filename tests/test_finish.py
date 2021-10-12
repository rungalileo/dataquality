import pytest

import dataquality
from dataquality import GalileoException


def test_finish_no_init():
    """
    Tests finish without an init call
    """
    dataquality.config.current_run_id = dataquality.config.current_project_id = None
    with pytest.raises(AssertionError):
        dataquality.finish()


def test_upload_thread_no_data():
    """
    Finish within a thread needs a model output dataframe. Without one an error should
    be thrown
    """
    dataquality.config.current_project_id = dataquality.config.current_run_id = "asdf"
    with pytest.raises(GalileoException):
        dataquality.upload(_in_thread=True, _model_output=None)
