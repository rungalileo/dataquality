from typing import Callable

from pytest import raises
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

from dataquality.integrations.seq2seq.hf import set_tokenizer
from tests.conftest import TestSessionVariables, tokenizer_T5, tokenizer_T5_not_auto


def test_set_tokenizer(
    set_test_config: Callable,
    cleanup_after_use: Callable,
    test_session_vars: TestSessionVariables,
) -> None:
    """
    Test that we can use either a PreTrainedFastTokenizer from HF or a Tokenizer from
    tokenizers, but not another type of tokenizer.
    """
    set_test_config(task_type="seq2seq")

    # Check that we can set the T5 auto tokenizer (of type PreTrainedTokenizerFast)
    assert isinstance(tokenizer_T5, PreTrainedTokenizerFast)
    set_tokenizer(tokenizer_T5)

    # Check that we can set a generic tokenizer (of type Tokenizer)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    assert isinstance(tokenizer, Tokenizer)
    set_tokenizer(tokenizer)

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
