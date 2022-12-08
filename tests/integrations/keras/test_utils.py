import tensorflow as tf

from dataquality.utils.keras import add_indices


def test_add_indices():
    """Test add_indices function convert the dataset into a zip dataset."""
    # Test with a dataset of tensors
    ds = tf.range(300)
    ds_idx = add_indices(ds)
    item = next(iter(ds_idx.batch(8)))
    assert len(item) == 2
    # Test with a dataset that is batched
    ds = tf.data.Dataset.from_tensor_slices(ds).batch(16)
    ds_idx = add_indices(ds)
    item = next(iter(ds_idx))
    assert len(item) == 2
