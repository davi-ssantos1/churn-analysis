"""Scale numeric columns from 0 to 1 and convert boolean columns to integer."""

from copy import deepcopy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _convert_boolean_to_int(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = deepcopy(dataframe)
    boolean_cols = dataframe.select_dtypes(include="boolean").columns
    dataframe[boolean_cols] = dataframe[boolean_cols].astype(dtype="Int64")
    return dataframe


def _minamax_scale(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = deepcopy(dataframe)
    num_cols = dataframe.select_dtypes(include="number").columns
    scaler = MinMaxScaler()
    dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])
    new_cols = {col: f"scaled_{col}" for col in num_cols}
    dataframe.rename(columns=new_cols)
    return dataframe


def convert_scale_dataframe(dataframe: pd.DataFrame, target: str) -> pd.DataFrame:
    """Scale numeric columns from 0 to 1 using sklearn MinMax scaler and convert boolean columns to integer (Int64).

    Args:
        dataframe: _description_
        target: _description_

    Returns:
        _description_
    """
    dataframe = _minamax_scale(dataframe=dataframe)
    dataframe = _convert_boolean_to_int(dataframe=dataframe)
    dataframe[target] = dataframe.pop(target)
    return dataframe
