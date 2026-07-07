"""Split data into train and test partitions."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split


def split_database(
    dataframe: pd.DataFrame, target: str
) -> Sequence[npt.NDArray[np.float64]]:
    """Split dataframe into x_train, x_test, y_train, y_test.

    Args:
        dataframe: Source data with X columns and a target column.
        target: Target column name.

    Returns:
        Tuple of numpy arrays (x_train, x_test, y_train, y_test).
    """
    x = dataframe.drop(columns=target).to_numpy()
    y = dataframe[target].to_numpy()

    split = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=123,
        stratify=y,
    )
    split = [arr.astype(np.float64) for arr in split]
    x_train, x_test, y_train, y_test = split
    return x_train, x_test, y_train, y_test
