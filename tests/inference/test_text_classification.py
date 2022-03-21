from unittest import mock

import pytest

import dataquality
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger


def test_set_split_inference() -> None:
    assert not dataquality.get_data_logger().logger_config.inference_logged
    dataquality.set_split("inference", "all-customers")
    assert dataquality.get_data_logger().logger_config.cur_split == "inference"
    assert (
        dataquality.get_data_logger().logger_config.cur_inference_name
        == "all-customers"
    )


def test_set_split_inference_missing_inference_name() -> None:
    with pytest.raises(AssertionError):
        dataquality.set_split("inference")


@mock.patch(
    "dataquality.loggers.model_logger.base_model_logger._save_hdf5_file",
    return_value="1234-abcd-5678",
)
@mock.patch(
    "dataquality.loggers.model_logger.base_model_logger.uuid4",
    return_value="1234-abcd-5678",
)
def test_write_model_output_inference(
    mock_uuid: mock.MagicMock, mock_save_file: mock.MagicMock
) -> None:
    inference_data = {
        "epoch": [None, None, None],
        "split": ["inference", "inference", "inference"],
        "inference_name": ["customers", "customers", "customers"],
    }
    logger = BaseGalileoModelLogger()
    logger.write_model_output(inference_data)

    # Assert _save_hdf5_file is called with correct args
    mock_save_file.assert_called_once_with(
        mock.ANY, "1234abcd5678.hdf5", inference_data
    )
    local_file = mock_save_file.call_args.args[0]
    assert local_file.endswith(
        f".galileo/logs/{dataquality.config.current_project_id}/"
        f"{dataquality.config.current_run_id}/inference/customers"
    )
