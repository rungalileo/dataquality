from typing import Callable

from pytest import raises
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from dataquality.integrations.seq2seq.hf import set_tokenizer
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
    set_tokenizer(tokenizer_T5)


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
    set_tokenizer(tokenizer)


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
    with raises(AssertionError) as context:
        set_tokenizer(tokenizer_T5_not_auto)
        assert (
            str(context.value)
            == "Tokenizer must be an instance of PreTrainedTokenizerFast"
        )
