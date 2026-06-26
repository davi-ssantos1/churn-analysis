from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd


def _get_cols_to_drop(
    dataframe: pd.DataFrame,
    target_dtype: Literal["number", "boolean"],
    threshold: float,
) -> list[str]:
    cols = dataframe.select_dtypes(include=target_dtype).columns
    corr = dataframe[cols].corr(method="pearson").abs().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    masked_corr = corr.where(cond=mask)
    cols_to_drop = [
        col_name
        for col_name in corr.columns
        if np.any(masked_corr[col_name] > threshold)
    ]
    return cols_to_drop


def drop_cols(
    dataframe: pd.DataFrame,
    target_dtypes: list[Literal["number", "boolean"]],
    threshold: float,
) -> pd.DataFrame:
    dataframe = deepcopy(dataframe)
    for dtype in target_dtypes:
        cols_to_drop = _get_cols_to_drop(
            dataframe=dataframe, target_dtype=dtype, threshold=threshold
        )
        dataframe = dataframe.drop(columns=cols_to_drop)

    return dataframe
