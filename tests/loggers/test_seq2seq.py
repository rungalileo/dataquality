from typing import Callable, Generator
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import vaex
from datasets import Dataset
from transformers import GenerationConfig, T5ForConditionalGeneration

import dataquality as dq
from dataquality.integrations.seq2seq.core import set_tokenizer, watch
from dataquality.loggers.data_logger.base_data_logger import DataSet
from dataquality.loggers.data_logger.seq2seq.seq2seq_base import Seq2SeqDataLogger
from dataquality.loggers.logger_config.seq2seq.seq2seq_base import seq2seq_logger_config
from dataquality.loggers.model_logger.seq2seq.seq2seq_base import Seq2SeqModelLogger
from dataquality.schemas.seq2seq import (
    TOP_K,
    BatchGenerationData,
)
from dataquality.schemas.seq2seq import Seq2SeqOutputCols as C
from dataquality.schemas.task_type import TaskType
from dataquality.utils.seq2seq.generation import (
    add_generated_output_to_df,
    generate_sample_output,
)
from dataquality.utils.thread_pool import ThreadPoolManager
from tests.conftest import (
    TestSessionVariables,
    model_T5,
    tokenizer,
    tokenizer_T5,
)


@pytest.mark.parametrize(
    "dataset",
    [
        pd.DataFrame(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        vaex.from_dict(
            {
                "summary": ["summary 1", "summary 2", "summary 3"],
                "title": ["title_1", "title_2", "title_3"],
                "my_id": [1, 2, 3],
            }
        ),
        Dataset.from_dict(
            dict(
                summary=["summary 1", "summary 2", "summary 3"],
                title=["title_1", "title_2", "title_3"],
                my_id=[1, 2, 3],
            )
        ),
    ],
)
def test_log_dataset_encoder_decoder(
    dataset: DataSet,
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")
    watch(tokenizer, "encoder_decoder")
    dq.log_dataset(dataset, text="summary", label="title", id="my_id", split="train")

    df = vaex.open(f"{test_session_vars.LOCATION}/input_data/training/data_0.arrow")
    expected_cols = [
        "id",
        "split",
        "text",
        "label",
        "token_label_positions",
        "token_label_offsets",
    ]
    assert sorted(df.get_column_names()) == sorted(expected_cols)
    assert df["text"].tolist() == ["summary 1", "summary 2", "summary 3"]
    assert df["label"].tolist() == ["title_1", "title_2", "title_3"]
    assert df["id"].tolist() == [1, 2, 3]
    assert df["split"].tolist() == ["training"] * 3


def test_log_dataset_no_tokenizer(set_test_config: Callable) -> None:
    set_test_config(task_type="seq2seq")
    df = pd.DataFrame(
        {
            "summary": ["summary 1", "summary 2", "summary 3"],
            "title": ["title_1", "title_2", "title_3"],
            "my_id": [1, 2, 3],
        }
    )
    # Note this functionality is tested fully by the Seq2Seq parent class
    logger = Seq2SeqDataLogger()
    with patch("dataquality.core.log.get_data_logger") as mock_method:
        mock_method.return_value = logger
        with pytest.raises(AssertionError) as e:
            dq.log_dataset(df, text="summary", label="title", id="my_id", split="train")
    assert str(e.value) == (
        "You must set your tokenizer before logging. "
        "Use `dq.integrations.seq2seq.core.set_tokenizer`"
    )


def test_log_model_outputs_encoder_decoder(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")

    tokenized_labels = [
        np.arange(10).tolist(),
        np.arange(18).tolist(),
        np.arange(20).tolist(),
        np.arange(4).tolist(),
    ]

    config = MagicMock()
    config.id_to_tokens = {}
    config.id_to_tokens["training"] = dict(zip(list(range(4)), tokenized_labels))
    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"
    config.tokenizer = mock_tokenizer
    config.tokenizer.decode = lambda x: "Fake"
    config.model_type = "encoder_decoder"

    batch_size = 4
    seq_len = 20
    vocab_size = 100

    logits = np.random.rand(batch_size, seq_len, vocab_size)
    # Set the locations of the "padded" tokens to 0 for the logits
    for idx, token_labels in enumerate(tokenized_labels):
        logits[idx, len(token_labels) :] = 0

    log_data = dict(
        ids=list(range(batch_size)),
        logits=logits,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    with patch("dataquality.core.log.get_model_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_model_outputs(**log_data)
    ThreadPoolManager.wait_for_threads()
    logger.check_for_logging_failures()
    output_data = vaex.open(f"{test_session_vars.LOCATION}/training/0/*.arrow")
    expected_cols = [
        "id",
        "token_logprobs",
        "top_logprobs",
        "split",
        "epoch",
    ]
    assert sorted(output_data.get_column_names()) == sorted(expected_cols)
    assert len(output_data) == 4

    token_logprobs = output_data["token_logprobs"].tolist()
    top_logprobs = output_data["top_logprobs"].tolist()

    for token_labels, sample_token_logprobs, sample_top_logprobs in zip(
        tokenized_labels, token_logprobs, top_logprobs
    ):
        assert (
            len(token_labels) == len(sample_token_logprobs) == len(sample_top_logprobs)
        )
        # Check all logprobs are < 0
        for token_logprob in sample_token_logprobs:
            assert token_logprob < 0

        # Check that we ignore <pad> token by checking that for each
        # token in sample_top_logprobs the top logprobs are not all equal.
        # Additionally check the general structure of top_logprobs
        for token_top_logprobs in sample_top_logprobs:
            assert len(token_top_logprobs) == TOP_K

            logprobs = [candidate[1] for candidate in token_top_logprobs]
            assert not np.allclose(logprobs[0], logprobs)

            assert np.alltrue(
                [candidate[0] == "Fake" for candidate in token_top_logprobs]
            )


def test_log_model_outputs_with_embs(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    set_test_config(task_type="seq2seq")

    tokenized_labels = [
        np.arange(10).tolist(),
        np.arange(18).tolist(),
        np.arange(20).tolist(),
        np.arange(4).tolist(),
    ]

    config = MagicMock()
    config.id_to_tokens = {}
    config.id_to_tokens["training"] = dict(zip(list(range(4)), tokenized_labels))
    mock_tokenizer = MagicMock()
    mock_tokenizer.padding_side = "right"
    config.tokenizer = mock_tokenizer
    config.tokenizer.decode = lambda x: "Fake"
    config.model_type = "encoder_decoder"

    batch_size = 4
    seq_len = 20
    vocab_size = 100

    logits = np.random.rand(batch_size, seq_len, vocab_size)
    # Set the locations of the "padded" tokens to 0 for the logits
    for idx, token_labels in enumerate(tokenized_labels):
        logits[idx, len(token_labels) :] = 0

    embs = np.random.rand(batch_size, 100)
    log_data = dict(
        ids=list(range(batch_size)),
        embs=embs,
        logits=logits,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    with patch("dataquality.core.log.get_model_logger") as mock_method:
        mock_method.return_value = logger
        dq.log_model_outputs(**log_data)
    ThreadPoolManager.wait_for_threads()
    logger.check_for_logging_failures()
    output_data = vaex.open(f"{test_session_vars.LOCATION}/training/0/*.arrow")
    expected_cols = [
        "id",
        "emb",
        "token_logprobs",
        "top_logprobs",
        "split",
        "epoch",
    ]
    assert sorted(output_data.get_column_names()) == sorted(expected_cols)
    assert len(output_data) == 4
    assert isinstance(output_data.emb.values, pa.ChunkedArray)


@patch("dataquality.utils.seq2seq.generation.generate_on_batch")
def test_add_generated_output_to_df(
    mock_generate_on_batch: Mock,
) -> None:
    """Test adding generation data to the df

    The main logic to test here is adding data to vaex. We also test
    the batched generation process.

    Things to Test:
        - Test that the correct df columns exist
        - Test that we can handle batched generation.

    Things to Mock
        - generate_on_batch: Create simple fake data for this
    """
    batch_size = 100

    # Create mock generation data per batch
    mock_generated_outputs = ["fake"] * batch_size

    fake_token_label_positions = [[{0, 1}, {1}, {0}, {2, 3}, {3}]]
    mock_token_label_positions = fake_token_label_positions * batch_size

    fake_token_label_offsets = [[(0, 1), (1, 20), (20, 21), (21, 22), (22, 23)]]
    mock_token_label_offsets = fake_token_label_offsets * batch_size

    num_tokens = 2
    mock_token_logprobs = [[-0.5, -0.1]] * batch_size
    mock_top_logprobs = [
        [[("A", -1), ("B", -2)] for _ in range(num_tokens)]
    ] * batch_size

    mock_generate_on_batch.return_value = BatchGenerationData(
        generated_outputs=mock_generated_outputs,
        generated_token_label_positions=mock_token_label_positions,
        generated_token_label_offsets=mock_token_label_offsets,
        generated_token_logprobs=mock_token_logprobs,
        generated_top_logprobs=mock_top_logprobs,
    )

    # Create fake df with vaex
    num_batches = 10
    df_size = batch_size * num_batches
    df = vaex.from_dict({"text": ["Fake Input"] * df_size})

    with patch(
        "dataquality.utils.seq2seq.generation.GENERATION_BATCH_SIZE", batch_size
    ):
        df = add_generated_output_to_df(df, Mock(), Mock(), 512, Mock())

    # Check the df columns!
    assert len(df) == df_size
    assert df[C.generated_output.value].tolist() == mock_generated_outputs * num_batches
    # Convert to correct type - this is because of the way vaex stores
    # sets and tuples - i.e. they need to be lists
    fake_token_label_positions = [[list(val) for val in fake_token_label_positions[0]]]
    assert (
        df[C.generated_token_label_positions.value].tolist()
        == fake_token_label_positions * num_batches * batch_size
    )
    fake_token_label_offsets = [[list(val) for val in fake_token_label_offsets[0]]]
    assert (
        df[C.generated_token_label_offsets.value].tolist()
        == fake_token_label_offsets * num_batches * batch_size
    )
    generated_token_logprobs = df[C.generated_token_logprobs.value].tolist()
    generated_top_logprobs = df[C.generated_top_logprobs.value].tolist()
    for logprobs, top_logprobs in zip(generated_token_logprobs, generated_top_logprobs):
        num_tokens = len(logprobs)
        assert num_tokens == len(top_logprobs)
        assert np.array_equal(logprobs, np.array([-0.5, -0.1]))
        assert top_logprobs == [[("A", -1), ("B", -2)] for _ in range(num_tokens)]


@patch("dataquality.utils.seq2seq.generation.process_sample_logprobs")
@patch("dataquality.utils.seq2seq.generation.get_top_logprob_indices")
def test_tokenize_input_provide_maxlength(
    mock_get_top_logprob_indices: Mock,
    mock_process_sample_logprobs: Mock,
    seq2seq_generated_output: torch.Tensor,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    # TODO comment!
    """
    Test that as we generate output and the user provided the max_input_tokens argument,
    the input is tokenized correctly to the length set by max_input_tokens.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_model = Mock(spec=T5ForConditionalGeneration)
    mock_model.device = "cpu"
    mock_model.return_value = Mock()
    mock_model.generate.return_value = seq2seq_generated_output
    mock_generation_config = Mock(spec=GenerationConfig)

    set_tokenizer(tokenizer_T5, "encoder_decoder", max_input_tokens=7)
    input_text = "a b c d e f g h i j"
    generate_sample_output(
        input_text,
        mock_model,
        tokenizer_T5,
        seq2seq_logger_config.max_input_tokens,
        mock_generation_config,
    )

    # Check that the input to generation was of length 7 (i.e, truncated)
    input_ids = mock_model.generate.call_args[1]["input_ids"]
    assert input_ids.shape == (1, 7)
    # Check that the methods after generation were called correctly
    mock_model.generate.assert_called_once_with(
        input_ids=input_ids, generation_config=mock_generation_config
    )
    mock_get_top_logprob_indices.assert_called_once()
    mock_process_sample_logprobs.assert_called_once()


@patch("dataquality.utils.seq2seq.generation.process_sample_logprobs")
@patch("dataquality.utils.seq2seq.generation.get_top_logprob_indices")
def test_tokenize_input_doesnt_provide_maxlength(
    mock_get_top_logprob_indices: Mock,
    mock_process_sample_logprobs: Mock,
    seq2seq_generated_output: torch.Tensor,
    set_test_config: Callable,
    cleanup_after_use: Generator,
) -> None:
    """
    Test that as we generate output and the user did not provide the max_input_tokens
    argument, the input is tokenized correctly to the length set by default in the
    tokenizer.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_model = Mock(spec=T5ForConditionalGeneration)
    mock_model.device = "cpu"
    mock_model.return_value = Mock()
    mock_model.generate.return_value = seq2seq_generated_output
    mock_generation_config = Mock(spec=GenerationConfig)

    set_tokenizer(tokenizer_T5, "encoder_decoder")
    input_text = "a b c d e f g h i j" * 100
    generate_sample_output(
        input_text,
        mock_model,
        tokenizer_T5,
        seq2seq_logger_config.max_input_tokens,
        mock_generation_config,
    )

    # Make sure that the input is large enough to require truncation
    assert len(input_text) > tokenizer_T5.model_max_length
    # Check that the input to generation was truncated (batch_size=1)
    input_ids = mock_model.generate.call_args[1]["input_ids"]
    assert input_ids.shape == (1, tokenizer_T5.model_max_length)
    # Check that the methods after generation were called correctly
    mock_model.generate.assert_called_once_with(
        input_ids=input_ids, generation_config=mock_generation_config
    )
    mock_get_top_logprob_indices.assert_called_once()
    mock_process_sample_logprobs.assert_called_once()


def test_tokenize_target_provide_maxlength_encoder_decoder(
    set_test_config: Callable, cleanup_after_use: Generator
) -> None:
    # TODO Update based on hf support for encoder-decoder vs. decoder-only
    """
    Test that the target is tokenized correctly to the length provided by the user in
    the max_target_tokens argument.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_generation_config = Mock(spec=GenerationConfig)
    watch(
        tokenizer_T5,
        "encoder_decoder",
        model_T5,
        mock_generation_config,
        max_target_tokens=7,
    )
    ds = Dataset.from_dict(
        {
            "id": [0, 1],
            "input": ["a b c d e f g h i j", "1"],
            "target": ["k l m n o p q r s t", "2"],
        }
    )
    dq.log_dataset(ds, text="input", label="target", split="train")

    assert set(seq2seq_logger_config.id_to_tokens["training"]) == {0, 1}
    assert len(seq2seq_logger_config.id_to_tokens["training"][0]) == 7
    # Check that it has two tokens: the token "2" + EOS token
    assert len(seq2seq_logger_config.id_to_tokens["training"][1]) == 2
    # Check that both sentences end with the same EOS token
    assert (
        seq2seq_logger_config.id_to_tokens["training"][0][-1]
        == seq2seq_logger_config.id_to_tokens["training"][1][-1]
    )


def test_tokenize_target_doesnt_provide_maxlength_encoder_decoder(
    set_test_config: Callable, cleanup_after_use: Generator
) -> None:
    # TODO Update based on hf support for encoder-decoder vs. decoder-only
    """
    Test that the target is tokenized correctly when the user does not provide a
    max_target_tokens argument, i.e., to the length set by default in the tokenizer.
    """
    set_test_config(task_type=TaskType.seq2seq)
    mock_generation_config = Mock(spec=GenerationConfig)
    # TODO Does using a real model here take a lot of time?
    #   should we just mock the model and add a max length?
    watch(tokenizer_T5, "encoder_decoder", model_T5, mock_generation_config)
    ds = Dataset.from_dict(
        {
            "id": [0, 1],
            "input": ["a b c d e f g h i j", "1"],
            "target": ["k l m n o p q r s t" * 100, "2"],
        }
    )
    dq.log_dataset(ds, text="input", label="target", split="train")

    assert set(seq2seq_logger_config.id_to_tokens["training"]) == {0, 1}
    # Make sure that the target is large enough to require truncation
    assert len(ds["target"][0]) > tokenizer_T5.model_max_length
    assert (
        len(seq2seq_logger_config.id_to_tokens["training"][0])
        == tokenizer_T5.model_max_length
    )
    # Check that it has two tokens: the token "2" + EOS token
    assert len(seq2seq_logger_config.id_to_tokens["training"][1]) == 2
    # Check that both sentences end with the same EOS token
    assert (
        seq2seq_logger_config.id_to_tokens["training"][0][-1]
        == seq2seq_logger_config.id_to_tokens["training"][1][-1]
    )


def test_calculate_cutoffs_encoder_decoder(
    set_test_config: Callable, cleanup_after_use: Generator
):
    """Test that calculate_cutoffs works correctly for both input/target"""
    set_test_config(task_type=TaskType.seq2seq)
    mock_model = Mock(spec=T5ForConditionalGeneration)
    mock_model.device = "cpu"
    mock_generation_config = Mock(spec=GenerationConfig)
    watch(
        tokenizer_T5,
        "encoder_decoder",
        mock_model,
        mock_generation_config,
        max_input_tokens=3,
        max_target_tokens=5,
    )

    input_1, input_2 = "dog dog dog done - tricked you", "bird"
    target_1, target_2 = "cat cat cat cat cat done", "cat"
    ds = Dataset.from_dict(
        {
            "id": [0, 1],
            "input": [input_1, input_2],
            "target": [target_1, target_2],
        }
    )
    data_logger = Seq2SeqDataLogger()
    data_logger.log_dataset(ds, text="input", label="target", split="training")
    in_frame_split = vaex.open(
        f"{data_logger.input_data_path}/training/*.{data_logger.INPUT_DATA_FILE_EXT}"
    )
    in_frame_split = data_logger.calculate_cutoffs(in_frame_split)
    input_offsets = in_frame_split["input_cutoff"].tolist()
    target_offsets = in_frame_split["target_cutoff"].tolist()

    assert len(input_offsets) == 2 == len(target_offsets)
    # The EOS token is always the last token, which we don't count for the cutoff point
    assert input_1[: input_offsets[0]] == "dog dog"
    assert target_1[: target_offsets[0]] == "cat cat cat cat"
    assert input_2[: input_offsets[1]] == "bird"
    assert target_2[: target_offsets[1]] == "cat"
