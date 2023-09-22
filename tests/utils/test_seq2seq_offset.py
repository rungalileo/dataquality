import vaex
import pyarrow as pa

def test_get_position_of_last_offset_target():
    """
    Test that get_position_of_last_offset_target returns the correct cut-off point for
    the target text string.
    """
    df = vaex.from_arrays(token_label_offsets=pa.array([1,2,3,3]))


    assert False



def test_get_position_of_last_offset_input():
    """
    Test that get_position_of_last_offset_input returns the correct cut-off point for
    the input text string.
    """
    assert False
