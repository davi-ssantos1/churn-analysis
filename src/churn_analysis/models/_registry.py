from typing import cast

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from churn_analysis.config import RANDOM_STATE
from churn_analysis.utils import ModelProtocol


def get_model(model_name: str) -> ModelProtocol:

    match model_name:
        case "xgboost":
            model = XGBClassifier(n_estimators=100, random_state=RANDOM_STATE)
            return cast(ModelProtocol, model)
        case "k_neighbors":
            model = KNeighborsClassifier()
            return model
        case "logistic_regression":
            model = LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
            return cast(ModelProtocol, model)
        case "random_forest":
            model = RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            )
            return cast(ModelProtocol, model)
        case _:
            raise ValueError(f"Model {model_name} was not found.")
