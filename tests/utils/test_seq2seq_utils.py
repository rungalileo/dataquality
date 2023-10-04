from dataclasses import dataclass
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pytest
import torch

from dataquality.exceptions import GalileoException
from dataquality.loggers.model_logger.seq2seq import Seq2SeqModelLogger
from dataquality.schemas.seq2seq import (
    TOP_K,
    AlignedTokenData,
    LogprobData,
    ModelGeneration,
)
from dataquality.utils.seq2seq import (
    remove_padding,
)
from dataquality.utils.seq2seq.generation import (
    generate_on_batch,
    generate_sample_output,
)
from dataquality.utils.seq2seq.logprobs import (
    get_top_logprob_indices,
    process_sample_logprobs,
)


@mock.patch("dataquality.utils.seq2seq.generation.align_tokens_to_character_spans")
@mock.patch("dataquality.utils.seq2seq.generation.generate_sample_output")
def test_generate_on_batch(
    mock_generate_sample_output: mock.Mock,
    mock_align_tokens_to_character_spans: mock.Mock,
) -> None:
    """Test generating over a batch of text Inputs

    In general, we have individually tested the functions used within
    `generate_on_batch`. So, here we are mainly testing the combined
    functionality of all of these.

    Things to Test:
        - Test that we are properly combining the per-sample info
        - Test the creation of the BatchGenerationData object

    Things to Mock:
        - generate_sample_output: This can be fairly simple. The
        one thing that we want to vary would be the length of things returned
        - tokenizer: decode can be quite simple + encode + max_input_tokens
        - model + generation_config: just to have as input params
        - align_tokens_to_character_spans: This is already tested so let's just
        mock the output!
    """

    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "Fake output"
    mock_tokenizer.return_value = {"offset_mapping": []}
    mock_max_input_tokens = 512

    # Mock the model
    mock_model = MagicMock()  # Don't actually need any functions for this

    # Mock generation_config
    mock_generation_config = MagicMock()

    # Mock the generation process
    def mock_generate_output() -> ModelGeneration:
        """Create simple dummy ModelGeneration"""
        # Generate fake outputs
        num_fake_tokens = np.random.randint(3, 10)
        gen_ids = np.arange(num_fake_tokens)
        gen_token_logprobs = np.zeros(num_fake_tokens) - 1
        gen_top_logprobs = [[("A", -1), ("B", -2)] for _ in range(num_fake_tokens)]
        gen_logprob_data = LogprobData(
            token_logprobs=gen_token_logprobs, top_logprobs=gen_top_logprobs
        )
        return ModelGeneration(gen_ids, gen_logprob_data)

    mock_generate_sample_output.return_value = mock_generate_output()

    # Mock aligned output for a single sample
    fake_token_label_offsets = [[(0, 1), (1, 20), (20, 21), (21, 22), (22, 23)]]
    fake_token_label_positions = [[{0, 1}, {1}, {0}, {2, 3}, {3}]]
    mock_align_tokens_to_character_spans.return_value = AlignedTokenData(
        token_label_offsets=fake_token_label_offsets,
        token_label_positions=fake_token_label_positions,
    )

    # Create fake df with vaex
    texts = pa.array(["Fake Input"] * 100)

    generated_data = generate_on_batch(
        texts, mock_model, mock_tokenizer, mock_max_input_tokens, mock_generation_config
    )

    # Make sure everything is in check!
    assert len(generated_data.generated_outputs) == 100
    assert generated_data.generated_outputs == ["Fake output"] * 100
    assert (
        generated_data.generated_token_label_positions
        == fake_token_label_positions * 100
    )
    assert (
        generated_data.generated_token_label_offsets == fake_token_label_offsets * 100
    )
    for logprobs, top_logprobs in zip(
        generated_data.generated_token_logprobs, generated_data.generated_top_logprobs
    ):
        num_tokens = len(logprobs)
        assert num_tokens == len(top_logprobs)
        assert np.array_equal(logprobs, np.zeros(num_tokens) - 1)
        assert top_logprobs == [[("A", -1), ("B", -2)] for _ in range(num_tokens)]


@mock.patch("dataquality.utils.seq2seq.generation.process_sample_logprobs")
@mock.patch("dataquality.utils.seq2seq.generation.get_top_logprob_indices")
def test_generate_sample_output(
    mock_get_top_logprob_indices: mock.Mock, mock_process_sample_logprobs: mock.Mock
) -> None:
    """Test the logic for generating over a single sample.

    Things to mock:
        - Anything model related
            - Assume .generate() works
            - Assume getting logits works
        - Mock the tokenizer
            - Mock the tokenize function
            - Mock the decode function
        - Mock get_top_logprob_indices - check inputs
        - Mock process_sample_logprobs since we test this seperately

    Things to test:
        - Check that the fake pad token is removed
        - Check that we have logprobs by checking input to process_sample_logprobs
        - Check that logprobs is correct shape to process_sample_logprobs
        - Check that gen_ids is correct shape to process_sample_logprobs
        - Check that we have the correct attributes in ModelGeneration
    """
    # Mock the tokenizer
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[0, 2, 3, 1]])}

    # Mock the model
    mock_model = mock.MagicMock()
    # Add a fake <pad> token to the generated ids
    mock_model.generate.return_value = torch.tensor([[1, 10, 20, 30]])

    # Mock the model forward function to return random logits
    # for a single batch element
    @dataclass
    class FakeOutput:
        logits: torch.tensor

    # Mock model output and model device
    mock_model.return_value = mock.MagicMock(logits=torch.rand((1, 3, 20)))
    mock_model.device = torch.device("cpu")

    # Mock generation_config
    mock_generation_config = mock.MagicMock()

    # Mock util helper function. Note we don't mock the return value of
    # mock_get_top_logprob_indices since we only care about its input
    fake_token_logprobs = np.array([-0.5, -0.25, -0.11])
    fake_top_logprob_data = [
        [("A", -0.1), ("B", -1)],
        [("A", -0.1), ("B", -1)],
        [("A", -0.1), ("B", -1)],
    ]
    mock_process_sample_logprobs.return_value = LogprobData(
        token_logprobs=fake_token_logprobs,
        top_logprobs=fake_top_logprob_data,
    )

    with mock.patch("torch.no_grad"):
        model_generation = generate_sample_output(
            "test str", mock_model, mock_tokenizer, 512, mock_generation_config
        )

    # Check logprobs
    logprobs = mock_get_top_logprob_indices.call_args.args[0]
    assert logprobs.shape == (3, 20)
    # Check that we infact have logprobs
    assert np.allclose(1.0, np.sum(np.exp(logprobs), axis=-1))

    # Check ModelGeneration
    # Check gen_ids - Make sure the <pad> token is removed!
    assert np.array_equal(model_generation.generated_ids, np.array([10, 20, 30]))
    assert np.array_equal(
        model_generation.generated_logprob_data.token_logprobs, fake_token_logprobs
    )
    assert model_generation.generated_logprob_data.top_logprobs == fake_top_logprob_data


@mock.patch("dataquality.utils.seq2seq.generation.get_top_logprob_indices")
def test_generate_sample_output_empty_sample(mock_get_top_logprob_indices: mock.Mock):
    """Test that we properly handle genearted sequences of length 1 - just [EOS]

    One tricky edge case if if the model immediately generates the EOS token. This is
    essentially equivalent to the model generating the empty string. Here we test
    to make sure nothing fails!

    Things to Mock:
        - Tokenizer `encode` and `decode`
        - model `generate`, `forward`, and `device`
        - generation_config
        - mock_get_top_logprob_indices
    """
    # Mock the tokenizer
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 0]])}
    mock_tokenizer.decode.return_value = "Fake"

    # Mock the model
    mock_model = mock.MagicMock()
    # Add a fake <pad> token to the generated ids
    mock_model.generate.return_value = torch.tensor([[0, 1]])

    # Mock the model forward function to return random logits
    # for a single batch element
    @dataclass
    class FakeOutput:
        logits: torch.tensor

    # Mock model output and model device - shape (bs=1, seq_len=1, 20)
    mock_model.return_value = mock.MagicMock(logits=torch.rand((1, 1, 20)))
    mock_model.device = torch.device("cpu")

    # Mock generation_config
    mock_generation_config = mock.MagicMock()

    # Mock top_k indices - shape (seq_len=1, 5)
    mock_get_top_logprob_indices.return_value = np.array([[1, 2, 3, 4, 5]])

    with mock.patch("torch.no_grad"):
        model_generation = generate_sample_output(
            "test str", mock_model, mock_tokenizer, 512, mock_generation_config
        )

    assert model_generation.generated_ids == np.array([1])
    assert len(model_generation.generated_logprob_data.token_logprobs) == 1


def test_model_logger_remove_padding() -> None:
    """Test _remove_padding and _retrieve_sample_labels

    Ensure that _remove_padding removes the correct tokens for each
    sample based on the `sample_labels` and the tokenzier padding direction.
    """
    tokenized_labels = [
        np.arange(10).tolist(),
        np.arange(18).tolist(),
        np.arange(20).tolist(),
        np.arange(4).tolist(),
    ]

    config = mock.MagicMock()
    config.id_to_tokens = {}
    config.id_to_tokens["training"] = dict(zip(list(range(4)), tokenized_labels))
    mock_tokenizer = mock.MagicMock()
    # First test removing from right padding
    config.tokenizer = mock_tokenizer

    batch_size = 4
    max_seq_len = 20
    vocab_size = 100

    logprobs = np.random.rand(batch_size, max_seq_len, vocab_size)
    # Set pad tokens on the right with -1
    for idx, token_labels in enumerate(tokenized_labels):
        logprobs[idx, len(token_labels) :] = -1
    # Create the top indices just using logits
    top_indices = logprobs[:, :, 5]

    # Note we don't differentiate between logits and logprobs for this test
    log_data = dict(
        ids=list(range(batch_size)),
        logits=logprobs,
        split="training",
        epoch=0,
    )
    logger = Seq2SeqModelLogger(**log_data)
    logger.logger_config = config
    for sample_id, (sample_logprobs, sample_top_indices) in enumerate(
        zip(logprobs, top_indices)
    ):
        sample_labels = logger._retrieve_sample_labels(sample_id, 100)
        # Test the retrieve samples method
        assert np.allclose(sample_labels, tokenized_labels[sample_id])

        no_pad_logprobs = remove_padding(sample_labels, "right", sample_logprobs)
        no_pad_top_indices = remove_padding(sample_labels, "right", sample_top_indices)
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0

    # Test padding on the 'left'
    logprobs = np.random.rand(batch_size, max_seq_len, vocab_size)
    # Set pad tokens on the left with -1
    for idx, token_labels in enumerate(tokenized_labels):
        logprobs[idx, : -len(token_labels)] = -1
    # Create the top indices just using logits
    top_indices = logprobs[:, :, 5]

    for sample_id, (sample_logprobs, sample_top_indices) in enumerate(
        zip(logprobs, top_indices)
    ):
        sample_labels = logger._retrieve_sample_labels(sample_id, 100)
        no_pad_logprobs = remove_padding(sample_labels, "left", sample_logprobs)
        no_pad_top_indices = remove_padding(sample_labels, "left", sample_top_indices)
        assert len(np.where(no_pad_logprobs == -1)[0]) == 0
        assert len(np.where(no_pad_top_indices == -1)[0]) == 0


def test_process_sample_logprobs():
    """Test process_sample_logprobs

    Ensure that the extracted label logprobs are correct
    and that the top_logprobs data is as expected.
    """
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.decode.return_value = "Fake"

    seq_len = 10
    vocab_size = 100

    fake_logprobs = np.random.rand(seq_len, vocab_size)
    fake_labels = np.arange(seq_len)
    fake_top_indices = np.tile(np.arange(TOP_K), (seq_len, 1))

    logprob_data = process_sample_logprobs(
        fake_logprobs, fake_labels, fake_top_indices, mock_tokenizer
    )

    # Check that the token_logprobs are correct
    token_logprobs = logprob_data.token_logprobs
    for i in range(len(token_logprobs)):
        assert token_logprobs[i] == fake_logprobs[i, fake_labels[i]]

    # Check that the top_logprobs are correct
    top_loprobs = logprob_data.top_logprobs
    assert len(top_loprobs) == seq_len
    assert len(top_loprobs[0]) == TOP_K
    for i, token_top_logprobs in enumerate(top_loprobs):
        pred_top_logprobs = [token[1] for token in token_top_logprobs]
        assert np.allclose(pred_top_logprobs, fake_logprobs[i, :TOP_K])


def test_process_sample_logprobs_seq_len_one():
    """Test process_sample_logprobs on a sequence of length one"""
    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.decode.return_value = "Fake"

    seq_len = 1
    vocab_size = 100

    fake_logprobs = np.random.rand(seq_len, vocab_size)
    fake_labels = np.arange(seq_len)
    fake_top_indices = np.tile(np.arange(TOP_K), (seq_len, 1))

    logprob_data = process_sample_logprobs(
        fake_logprobs, fake_labels, fake_top_indices, mock_tokenizer
    )

    # Check that the token_logprobs are correct
    assert logprob_data.token_logprobs.shape == (1,)

    # Check that the top_logprobs are correct
    top_loprobs = logprob_data.top_logprobs
    assert len(top_loprobs) == seq_len
    assert len(top_loprobs[0]) == TOP_K


def test_process_sample_logprobs_incorrect_shape():
    """Test process_sample_logprobs with incorrect label shape"""
    mock_tokenizer = mock.MagicMock()
    seq_len = 10
    vocab_size = 100
    fake_logprobs = np.zeros((seq_len, vocab_size))
    fake_top_indices = np.zeros((seq_len, 5))

    # We expect labels to have shape (seq_len,) when passing
    # to process_sample_logprobs
    incorrect_labels = np.zeros((seq_len, 1))

    with pytest.raises(GalileoException) as excinfo:
        _, _ = process_sample_logprobs(
            fake_logprobs, incorrect_labels, fake_top_indices, mock_tokenizer
        )

    assert (
        "Invalid shape (10, 1), process_sample_logprobs"
        " expects sample_labels to be a 1D array" == str(excinfo.value)
    )


def test_get_top_logprob_indices() -> None:
    """
    Test getting the top 5 logprobs with two different tensor shapes!
        - [seq_len, vc]
        - [bs, seq_len, vc]

    Use arange so that we can expect the exact result!
    """
    batch_size = 4
    seq_len = 10
    vocab_size = 100

    # Test logprobs shape - [seq_len, vocab_size]
    logprobs = np.random.rand(seq_len, vocab_size)
    copy_logprobs = logprobs.copy()
    top_logprob_indices = get_top_logprob_indices(logprobs)
    # Make sure we don't modify the logprobs
    assert np.allclose(logprobs, copy_logprobs)

    # Manually argsort - i.e. the slower way to do this!
    manual_top_logprob_indices = np.argsort(logprobs, axis=-1)[:, -TOP_K:]
    # Note top_logprob_indices is not guaranteed to be sorted
    for token, gt_token in zip(top_logprob_indices, manual_top_logprob_indices):
        token = set(list(token))
        gt_token = set(list(gt_token))
        assert token == gt_token

    # Test logprobs shape - [batch_size, seq_len, vocab_size]
    # Use a simple constructed case where each token has the same "logprobs"
    logprobs = np.tile(np.arange(vocab_size), (batch_size, seq_len, 1))
    top_logprob_indices = get_top_logprob_indices(logprobs)

    assert top_logprob_indices.shape == (batch_size, seq_len, TOP_K)
    # Manually construct desired output based on how partition works
    gt_top_logprob_indices = np.tile(
        np.array([98, 99, 97, 96, 95]), (batch_size, seq_len, 1)
    )
    assert np.allclose(top_logprob_indices, gt_top_logprob_indices)
