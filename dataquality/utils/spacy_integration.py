from typing import Any
from dataquality.exceptions import GalileoException


def validate_obj(an_object: Any, check_type: Any, has_attr: str) -> None:
    if not isinstance(an_object, check_type):
        raise GalileoException(
            f"Expected a {check_type}. Received {str(type(an_object))}"
        )

    if not hasattr(an_object, has_attr):
        raise GalileoException(f"Your {check_type} must have a {has_attr} attribute")
