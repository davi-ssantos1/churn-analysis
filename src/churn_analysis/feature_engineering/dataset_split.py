from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import train_test_split


def split_database(
    dataframe: pd.DataFrame, target: str
) -> Sequence[npt.NDArray[np.float64]]:
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
