from typing import Callable

import pytest
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from dataquality.exceptions import GalileoException
from dataquality.integrations.seq2seq.core import set_tokenizer, watch
from dataquality.schemas.seq2seq import Seq2SeqModelType
from tests.conftest import TestSessionVariables, tokenizer_T5, tokenizer_T5_not_auto


def test_set_tokenizer_PreTrainedFastTokenizer(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we can use a PreTrainedFastTokenizer from Transformers"""
    set_test_config(task_type="seq2seq")

    # Check that we can set the T5 auto tokenizer (of type PreTrainedTokenizerFast)
    assert isinstance(tokenizer_T5, PreTrainedTokenizerFast)
    set_tokenizer(tokenizer_T5, "encoder_decoder")


def test_set_tokenizer_Tokenizer(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we can use a Tokenizer from tokenizers"""
    set_test_config(task_type="seq2seq")

    # Check that we can set a generic tokenizer (of type Tokenizer)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    assert isinstance(tokenizer, Tokenizer)
    set_tokenizer(tokenizer, "encoder_decoder")


def test_set_tokenizer_other(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test that we can't use a tokenizer that is not a PreTrainedFastTokenizer from
    Transformers or a Tokenizer from tokenizers.
    """
    set_test_config(task_type="seq2seq")

    # Check that the T5 tokenizer of another type is not working for now
    tokenizer = tokenizer_T5_not_auto
    assert not (
        isinstance(tokenizer, PreTrainedTokenizerFast)
        or isinstance(tokenizer, Tokenizer)
    )
    with pytest.raises(ValueError) as e:
        set_tokenizer(tokenizer_T5_not_auto, "encoder_decoder")
    assert str(e.value) == (
        "The tokenizer must be an instance of PreTrainedTokenizerFast " "or Tokenizer"
    )


def test_watch_invalid_task_type(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we can't watch for non-seq2seq tasks"""
    set_test_config(task_type="text_classification")
    with pytest.raises(AssertionError) as e:
        watch(tokenizer_T5, "encoder_decoder")
    assert str(e.value) == (
        "This method is only supported for seq2seq tasks. "
        "Make sure to set the task type with dq.init()"
    )


def test_watch_invalid_model_type(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we can't watch without an appropriate model_type"""
    set_test_config(task_type="seq2seq")
    with pytest.raises(ValueError) as e:
        watch(tokenizer_T5, "invalid_model_type")
    assert str(e.value) == (
        f"model_type must be one of {Seq2SeqModelType.members()}, "
        "got invalid_model_type"
    )


def test_watch_response_template_required(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we need response template to be passed for decoder-only"""
    set_test_config(task_type="seq2seq")
    with pytest.raises(GalileoException) as e:
        watch(tokenizer_T5, "decoder_only")
    assert str(e.value) == (
        "You must specify a `response_template` when using Decoder-Only models."
        " This is necessary to internally isolate the target response tokens."
    )


def test_watch_response_template_not_tokenized(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """Test that we need response template to be tokenized (for decoder-only)"""
    set_test_config(task_type="seq2seq")
    with pytest.raises(GalileoException) as e:
        watch(tokenizer_T5, "decoder_only", response_template="###\nRESPONSE:\n")
    assert (
        str(e.value)
        == "The response template must already be tokenized and be a list of ints."
    )
