from churn_analysis.config import RAW_DATA_PATH
from churn_analysis.etl import replace_missing_values_in_chunks, validate_data


def main() -> int:
    for df_chunk in replace_missing_values_in_chunks(
        raw_data_path=RAW_DATA_PATH,
        extraction_func=validate_data,
        col_title="total_charges",
    ):
        print(df_chunk)
    return 0


if __name__ == "__main__":
    SystemExit(main())
