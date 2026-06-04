from churn_analysis.config import DB_PATH, DB_TABLE_NAME, RAW_DATA_PATH
from churn_analysis.etl import extract_transform_load


def main() -> int:
    extract_transform_load(
        db_path=DB_PATH, db_table_name=DB_TABLE_NAME, raw_data_path=RAW_DATA_PATH
    )

    return 0


if __name__ == "__main__":
    SystemExit(main())
