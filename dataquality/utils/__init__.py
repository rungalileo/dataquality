try:
    from IPython import get_ipython

    if get_ipython():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm

tqdm = tqdm

__all__ = ["tqdm"]
