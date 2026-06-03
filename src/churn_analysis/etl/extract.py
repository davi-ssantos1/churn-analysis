from collections.abc import Generator
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from churn_analysis.etl.schemas import CustomerRecord


def validate_data(raw_data_path: Path) -> Generator[pd.DataFrame, None, None]:
    """
    Extract the CSV file in chunks, validade and yield as DataFrames.

    Architecture Note:
    While this specific 7000 row dataset could be fitted in RAM, this extraction pipeline
    implements a chunked generator pattern to demonstrate memory safe for an professional ETL.
    """

    csv_reader = pd.read_csv(raw_data_path, chunksize=1000)
    for df_chunk_index, df_chunk in enumerate(csv_reader):
        clean_chunks = []
        records = df_chunk.to_dict(orient="records")

        for row_index, row in enumerate(records):
            try:
                custom_clean_record = CustomerRecord(**row)
                dict_clean_record = custom_clean_record.model_dump()
                clean_chunks.append(dict_clean_record)
            except ValidationError:
                raise Exception(
                    f"Corrupted data from {df_chunk_index * 1000 + row_index + 1} row.\nRow: {row}"
                ) from None

        yield pd.DataFrame(clean_chunks)
