"""Database connection and data extraction utilities."""

import sqlite3

import pandas as pd

from churn_analysis.config import DB_PATH, DB_TABLE_NAME, SCHEMA_MAP, TARGET_COLUMN_NAME


def get_dataframe_from_dbwarehouse(with_target_column: bool) -> pd.DataFrame:
    """Get and format the dataset from the SQLite data warehouse.

    Args:
        with_target_column: Boolean flag indicanting whether to keep the target label column.

    Returns:
        A Pandas Dataframe mapped to the configured schema, with the customer_id column removed.
    """
    query = f"SELECT * FROM {DB_TABLE_NAME}"
    with sqlite3.connect(DB_PATH) as conn:
        dataframe = pd.read_sql_query(sql=query, con=conn, dtype=SCHEMA_MAP)
    dataframe = dataframe.drop(columns="customer_id")
    if with_target_column:
        return dataframe
    return dataframe.drop(columns=TARGET_COLUMN_NAME)
