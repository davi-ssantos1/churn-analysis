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
    query = f"SELECT * FROM {db_table_name}"
    df = execute_query(query=query, db_path=db_path, schema_map=schema_map)
    df = df.drop(columns=id_column)
    df = pd.get_dummies(
        data=df, columns=df.select_dtypes(include="string").columns, drop_first=True
    )
    return df
