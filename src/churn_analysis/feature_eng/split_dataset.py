"""Split data into train and test partitions."""

from collections.abc import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from churn_analysis.config import RANDOM_STATE, TARGET_COLUMN_NAME
from churn_analysis.utils import get_dataframe_from_dbwarehouse


def get_train_test_split() -> Sequence[pd.DataFrame | pd.Series]:
    """Split dataframe into x_train, x_test, y_train, y_test.

    Returns:
        Tuple of numpy arrays (x_train, x_test, y_train, y_test).
    """
    df = get_dataframe_from_dbwarehouse(with_target_column=True)
    x = df.drop(columns=TARGET_COLUMN_NAME)
    y = df[TARGET_COLUMN_NAME]

    split = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_train, x_test, y_train, y_test = split
    return x_train, x_test, y_train, y_test
