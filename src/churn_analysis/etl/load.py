import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def load_in_chunks(
    db_path: Path, table_name: str, data_stream: Iterable[pd.DataFrame]
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")

        for df_chunk in data_stream:
            df_chunk.to_sql(
                name=table_name,
                con=conn,
                if_exists="append",
                index=False,
            )
