from typing import List, Union

import numpy as np
import pyarrow as pa
import vaex
from vaex.dataframe import DataFrame


def expand_df(df: DataFrame, col: str = "emb") -> DataFrame:
    """Expands a dataframe with a single column list into many columns.

    ex: a dataframe with "emb" column containing a list of 10 floats

    #    id    emb
    0    0     '[0.936133472790654, 0.5483493321472575, 0.48622...
    1    1     '[0.7135356216870609, 0.21236254961668966, 0.465...
    2    2     '[0.7181627347404738, 0.9197334314857543, 0.3790...
    ...  ...   ...
    will be returned as a new dataframe with 10 new columns, and the original column
    removed:

    #    id    emb_0                 emb_1                emb_2               ...
    0    0     0.9993798461057117    0.02304161614313227  0.0859653860282843  ...
    1    1     0.9303099983067433    0.7801853589012353   0.4132189907682057  ...
    2    2     0.6392906559734074    0.9060376615716177   0.9146933280987043  ...
    ...  ...   ...                   ...                  ...

    NOTE: The dataframe passed in is modified and returned, a copy IS NOT made

    :param df: The Vaex DataFrame
    :param col: The column to expand
    """
    @vaex.register_function()
    def expand(col: List[List[Union[float, pa.float64]]], ind: int):
        is_pa = not isinstance(col[0][0], float)
        get_float = lambda v: v.as_py() if is_pa else float(v)
        t = [get_float(row[ind]) for row in col]
        return pa.array(t)

    num_c = len(df[col][:1].values[0])
    for i in range(num_c):
        df[f"{col}_{i}"] = df[col].expand(i)
    cols = [c for c in df.get_column_names() if c != col]
    return df[cols]

