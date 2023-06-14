from typing import Dict

import pyarrow as pa

def save_arrow_file(location: str, file_name: str, data: Dict) -> None:
    """
    Helper function to save a dictionary as an hdf5 file that can be read by vaex
    """
    if not os.path.isdir(location):
        with lock:
            os.makedirs(location, exist_ok=True 
