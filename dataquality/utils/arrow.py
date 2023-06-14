import os
from typing import Dict

import pyarrow as pa

from dataquality.utils.thread_pool import lock


def save_arrow_file(location: str, file_name: str, data: Dict) -> None:
    """
    Helper function to save a dictionary as an hdf5 file that can be read by vaex
    """
    if not os.path.isdir(location):
        with lock:
            os.makedirs(location, exist_ok=True )
    # TODO: Create arrow file and save
