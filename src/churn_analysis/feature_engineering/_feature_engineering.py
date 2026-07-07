"""Execute feature engineering pipeline."""

from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from churn_analysis.feature_engineering.convert_scale_columns import (
    convert_scale_dataframe,
)
from churn_analysis.feature_engineering.dataset_split import split_database
from churn_analysis.feature_engineering.drop_columns import drop_cols
from churn_analysis.feature_engineering.get_dummies_str import get_dummed_dataframe


def execute_feat_engineering(
    id_column: str,
    db_path: Path,
    db_table_name: str,
    schema_map: Mapping[Hashable, str],
) -> Sequence[npt.NDArray[np.float64]]:
    """Execute the featue engineering pipeline for statistical models.

    This process one-hot encodes categorical columns, drops highly correlated
    columns, scales numeric columns between 0.0 and 1.0, transform boolean columns
    into integers, and split the database into train and test partitions.

    Args:
        id_column: Primary key column name.
        db_path: Absolute or relative path to the database file.
        db_table_name: Name of the target table to create and populate.
        schema_map: Contract for SQL query datatypes.

    Returns:
        Tuple of numpy arrays (x_train, x_test, y_train, y_test).
    """
    df = get_dummed_dataframe(
        id_column=id_column,
        db_path=db_path,
        db_table_name=db_table_name,
        schema_map=schema_map,
    )
    df = drop_cols(dataframe=df, target_dtypes=["number", "boolean"], threshold=0.8)
    df = convert_scale_dataframe(dataframe=df, target="churn")
    x_train, x_test, y_train, y_test = split_database(dataframe=df, target="churn")
    return x_train, x_test, y_train, y_test
