from pathlib import Path

from churn_analysis.etl.extract import validate_data
from churn_analysis.etl.load import load_in_chunks
from churn_analysis.etl.transform import replace_missing_values_in_chunks


def extract_transform_load(
    raw_data_path: Path, db_path: Path, db_table_name: str
) -> None:
    clean_data_stream = replace_missing_values_in_chunks(
        raw_data_path=raw_data_path,
        extraction_func=validate_data,
        col_title="total_charges",
    )
    load_in_chunks(
        db_path=db_path,
        table_name=db_table_name,
        data_stream=clean_data_stream,
    )
