from collections.abc import Hashable, Mapping
from pathlib import Path
from types import MappingProxyType

# Path files to be used
ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent
RAW_DATA_PATH = ROOT_PATH / "data" / "raw_data" / r"Telco-Customer-Churn.csv"
DB_PATH = ROOT_PATH / "data" / "processed_data" / "churn_warehouse.db"
DB_TABLE_NAME = "customer_churn_records"
SCHEMA_MAP: Mapping[Hashable, str] = MappingProxyType(
    {
        "customer_id": "string",
        "gender": "string",
        "senior_citizen": "boolean",
        "partner": "boolean",
        "dependents": "boolean",
        "tenure": "Int64",
        "phone_service": "boolean",
        "multiple_lines": "string",
        "internet_services": "string",
        "online_security": "boolean",
        "online_backup": "boolean",
        "device_protection": "boolean",
        "tech_support": "boolean",
        "streaming_tv": "boolean",
        "streaming_movies": "boolean",
        "contract": "string",
        "paper_less_billing": "boolean",
        "payment_method": "string",
        "monthly_charges": "Float64",
        "total_charges": "Float64",
        "churn": "boolean",
    }
)
MODEL_NAMES = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "k_nearest_neighbors",
)
RANDOM_STATE = 123
