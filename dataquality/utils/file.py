import os


def get_file_extension(path: str) -> str:
    """Returns the file extension"""
    return os.path.splitext(path)[-1]
