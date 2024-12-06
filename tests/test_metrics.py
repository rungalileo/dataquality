from unittest.mock import Mock
from dataquality import metrics
import vaex


def test_get_edited_dataframe_all_edits(mocker):
    reviewed_only = False

    project_id = "project_id"
    project_name = "project_name"
    run_id = "run_id"
    run_name = "run_name"
    task_type = "task_type"
    file_type = Mock()
    uuid = "test-uuid"
    inference_name = ""
    hf_format = False
    tagging_schema = Mock()
    as_pandas = True
    include_embs = False
    include_data_embs = False
    include_probs = False
    include_token_indices = False

    test_df = vaex.from_dict({
        "id": range(0, 10),
        "confidence": [0.7] * 10,
        "is_drifted": [False] * 7 + [True] * 3,
        "reviewers": [[]] * 7 + [["review1"]] * 3,
    })

    api_mock = mocker.patch.object(metrics, "api_client")
    split = Mock()
    conform_split_mock = mocker.patch("dataquality.metrics.conform_split")
    split_mock = conform_split_mock.return_value

    api_mock._get_project_run_id.return_value = [project_id, run_id]
    api_mock.get_task_type.return_value = task_type

    mocker.patch("dataquality.metrics.uuid4", return_value = uuid)
    mocker.patch("dataquality.vaex.open", return_value = test_df)

    _process_exported_dataframe_mock = Mock("dataquality.metrics._process_exported_dataframe")

    response = metrics.get_edited_dataframe(project_name, run_name, split, inference_name, file_type, include_embs, include_probs, include_token_indices, hf_format, tagging_schema, reviewed_only, as_pandas, include_data_embs)

    assert response == _process_exported_dataframe_mock.return_value
    assert conform_split_mock.assert_called_once_with(split)
    assert api_mock._get_project_run_id.assert_called_once_with(project_name, run_name)
    assert api_mock.get_task_type.assert_called_once_with(project_id, run_id)

    assert api_mock.export_edits.assert_called_once_with(
        project_name,
        run_name,
        split_mock,
        inference_name = inference_name,
        file_name = f"/tmp/{uuid}-data.{file_type}",
        hf_format = hf_format,
        tagging_schema = tagging_schema,
    )

    assert _process_exported_dataframe_mock.assert_called_once_with(
        test_df,
        project_name,
        run_name,
        split,
        task_type,
        inference_name,
        include_embs,
        include_probs,
        include_token_indices,
        hf_format,
        as_pandas,
        include_data_embs,
    )


def test_get_edited_dataframe_reviewed_only_edits(mocker):
    reviewed_only = True

    project_id = "project_id"
    project_name = "project_name"
    run_id = "run_id"
    run_name = "run_name"
    task_type = "task_type"
    file_type = Mock()
    uuid = "test-uuid"
    inference_name = ""
    hf_format = False
    tagging_schema = Mock()
    as_pandas = True
    include_embs = False
    include_data_embs = False
    include_probs = False
    include_token_indices = False

    test_df = vaex.from_dict({
        "id": range(0, 10),
        "confidence": [0.7] * 10,
        "is_drifted": [False] * 7 + [True] * 3,
        "reviewers": [[]] * 7 + [["review1"]] * 3,
    })

    expected_df = vaex.from_dict({
        "id": range(7, 10),
        "confidence": [0.7] * 3,
        "is_drifted": [True] * 3,
        "reviewers": [["review1"]] * 3,
    })

    api_mock = mocker.patch.object(metrics, "api_client")
    split = Mock()
    conform_split_mock = mocker.patch("dataquality.metrics.conform_split")
    split_mock = conform_split_mock.return_value

    api_mock._get_project_run_id.return_value = [project_id, run_id]
    api_mock.get_task_type.return_value = task_type

    mocker.patch("dataquality.metrics.uuid4", return_value = uuid)
    mocker.patch("dataquality.vaex.open", return_value = test_df)

    _process_exported_dataframe_mock = Mock("dataquality.metrics._process_exported_dataframe")

    response = metrics.get_edited_dataframe(project_name, run_name, split, inference_name, file_type, include_embs, include_probs, include_token_indices, hf_format, tagging_schema, reviewed_only, as_pandas, include_data_embs)

    assert response == _process_exported_dataframe_mock.return_value
    assert conform_split_mock.assert_called_once_with(split)
    assert api_mock._get_project_run_id.assert_called_once_with(project_name, run_name)
    assert api_mock.get_task_type.assert_called_once_with(project_id, run_id)

    assert api_mock.export_edits.assert_called_once_with(
        project_name,
        run_name,
        split_mock,
        inference_name = inference_name,
        file_name = f"/tmp/{uuid}-data.{file_type}",
        hf_format = hf_format,
        tagging_schema = tagging_schema,
    )

    assert _process_exported_dataframe_mock.assert_called_once_with(
        expected_df,
        project_name,
        run_name,
        split,
        task_type,
        inference_name,
        include_embs,
        include_probs,
        include_token_indices,
        hf_format,
        as_pandas,
        include_data_embs,
    )
