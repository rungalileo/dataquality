from dataquality.utils.name import ADJECTIVES, ANIMALS, COLORS, random_name


def test_name() -> None:
    adj, c, anm = random_name().split("_")
    assert adj in ADJECTIVES
    assert c in COLORS
    assert anm in ANIMALS
