"""'Model pipeline selection module."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from churn_analysis.config import RANDOM_STATE


def get_model(
    model_name: str, preprocessor: ColumnTransformer, apply_smote: bool = False
) -> Pipeline:
    """Create a model instance based on the provided name.

    Args:
        model_name: Name of the model to be selected.
        preprocessor: Configured Scikit-Learn ColumnTransformer for feature engineering.
        apply_smote: Set if SMOTE oversampling will be applied in the model training step.

    Raises:
        ValueError: If module_name is not recognized.

    Returns:
        An Imblearn Pipeline containing the preprocessor, optional sampler and classifier.
    """
    steps = [("preprocessor", preprocessor)]
    if apply_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))

    match model_name:
        case "xgboost":
            model = XGBClassifier(scale_pos_weight=2.76, random_state=RANDOM_STATE)
        case "k_neighbors":
            model = KNeighborsClassifier()
        case "logistic_regression":
            model = LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        case "random_forest":
            model = RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            )
        case _:
            raise ValueError(f"Model {model_name} was not found.")
    steps.append(("classifier", model))
    return Pipeline(steps=steps)
