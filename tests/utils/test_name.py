import pytest

from dataquality.exceptions import GalileoException
from dataquality.utils.name import (
    ADJECTIVES,
    ANIMALS,
    COLORS,
    random_name,
    validate_name,
)


@pytest.mark.parametrize(
    "name",
    [
        "onlyletters",
        "als0numb3rs",
        "and even spaces",
        "dashes-are-fine",
        "so_are_these_things",
    ],
)
def test_validate_name(name: str) -> None:
    assert name == validate_name(name)


def test_validate_name_assert_random() -> None:
    """Generate 100 random names and assert that they are valid"""
    for _ in range(100):
        validate_name(None, assign_random=True)


@pytest.mark.parametrize(
    "name,badchars",
    [
        ("test!", "['!']"),
        ("feature/name", "['/']"),
        ("this,should,fail", "[',', ',']"),
    ],
)
def test_validate_name_fails(name: str, badchars: str) -> None:
    with pytest.raises(GalileoException) as e:
        validate_name(name)

    assert str(e.value) == (
        "Only letters, numbers, whitespace, - and _ are allowed in a project "
        f"or run name. Remove the following characters: {badchars}"
    )


def test_validate_name_missing_name() -> None:
    with pytest.raises(GalileoException) as e:
        validate_name(None)

    assert str(e.value) == (
        "Name is required. Set assign_random to True to generate a random name"
    )


def test_name() -> None:
    adj, c, anm = random_name().split("_")
    assert adj in ADJECTIVES
    assert c in COLORS
    assert anm in ANIMALS
