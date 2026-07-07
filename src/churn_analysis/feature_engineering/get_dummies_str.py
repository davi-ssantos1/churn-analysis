"""Apply one hot encoding in columns with string data type."""

from collections.abc import Hashable, Mapping
from pathlib import Path

import pandas as pd

from churn_analysis.utils import execute_query


def get_dummed_dataframe(
    id_column: str,
    db_path: Path,
    db_table_name: str,
    schema_map: Mapping[Hashable, str],
) -> pd.DataFrame:
    """Get the pandas DataFrame from a database and apply pandas.get_dummies function with drop_first=True to string columns.

    Args:
        id_column: The table's primary key.
        db_path: The absolute or relative path to the database.
        db_table_name: The name of the table to be acessed
        schema_map: Used to set feature data type when querying the database into a pandas DataFrame

    Returns:
        Dataframe with one hot encoded for string features. It delete the source columns used to get one hot encoding.
    """
    query = f"SELECT * FROM {db_table_name}"
    df = execute_query(query=query, db_path=db_path, schema_map=schema_map)
    df = df.drop(columns=id_column)
    df = pd.get_dummies(
        data=df, columns=df.select_dtypes(include="string").columns, drop_first=True
    )
    return df
