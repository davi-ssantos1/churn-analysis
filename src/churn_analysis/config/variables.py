from pathlib import Path

# Path files to be used
ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw_data" / r"Telco-Customer-Churn.csv"
DB_PATH = ROOT_PATH / "data" / "processed_data" / "churn_warehouse.db"
DB_TABLE_NAME = "customer_churn_records"
