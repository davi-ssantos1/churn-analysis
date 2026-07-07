"""Load data to a SQLite database for the churn analysis pipeline."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def load_in_chunks(
    db_path: Path, table_name: str, data_stream: Iterable[pd.DataFrame]
) -> None:
    """Load dataframe chunks from a data stream into a SQLite database.

    Args:
        db_path: Absolute or relative path to the data base file.
        table_name: Name of the target table to create and populate.
        data_stream: Iterable stream of dataframe chuncks to be loaded.
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        for df_chunk in data_stream:
            df_chunk.to_sql(
                name=table_name,
                con=conn,
                if_exists="append",
                index=False,
            )
