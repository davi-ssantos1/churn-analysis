"""Execute churn analysis pipeline."""

from churn_analysis.config import DB_PATH, DB_TABLE_NAME, RAW_DATA_PATH
from churn_analysis.etl import extract_transform_load
from churn_analysis.feature_eng import get_train_test_split
from churn_analysis.models import ModelsPipeline


def main() -> int:
    """Churn analysis entrypoint.

    Returns:
        The integer value that will be consumed by SystemExit.
    """
    extract_transform_load(
        db_path=DB_PATH, db_table_name=DB_TABLE_NAME, raw_data_path=RAW_DATA_PATH
    )

    x_train, x_test, y_train, y_test = get_train_test_split()

    models_pipeline = ModelsPipeline(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )
    best_model_name, flatted_metrics = models_pipeline.run()
    print(f"Model name: {best_model_name}\nFlatted metrics: {flatted_metrics}")
    return 0


if __name__ == "__main__":
    SystemExit(main())
