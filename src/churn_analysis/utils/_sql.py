import sqlite3
from collections.abc import Hashable, Mapping
from pathlib import Path

import pandas as pd


def execute_query(
    query: str, db_path: Path, schema_map: Mapping[Hashable, str]
) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        result = pd.read_sql_query(sql=query, con=conn, dtype=schema_map)
    return result
