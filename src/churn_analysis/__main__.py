"""Execute churn analysis pipeline."""

from churn_analysis.config import DB_PATH, DB_TABLE_NAME, RAW_DATA_PATH, SCHEMA_MAP
from churn_analysis.etl import extract_transform_load
from churn_analysis.feature_eng import execute_feat_engineering


def main() -> int:
    """Churn analysis entrypoint.

    Returns:
        The integer value that will be consumed by SystemExit.
    """
    extract_transform_load(
        db_path=DB_PATH, db_table_name=DB_TABLE_NAME, raw_data_path=RAW_DATA_PATH
    )
    x_train, x_test, y_train, y_yest = execute_feat_engineering(
        id_column="customer_id",
        db_path=DB_PATH,
        db_table_name=DB_TABLE_NAME,
        schema_map=SCHEMA_MAP,
    )

    return 0


if __name__ == "__main__":
    SystemExit(main())
