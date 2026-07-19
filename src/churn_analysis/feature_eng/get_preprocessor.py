"""Data preprocessor module."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as Sklearn_Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from churn_analysis.config import ESTIMATOR_MODELS

CORRELATION_THRESHOLD = 0.8


def _get_numeric_cols_to_drop(
    dataframe: pd.DataFrame,
) -> list[str]:
    cols = dataframe.select_dtypes(include="number").columns
    corr = dataframe[cols].corr(method="pearson").abs().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    masked_corr = corr.where(cond=mask)
    cols_to_drop = [
        col_name
        for col_name in corr.columns
        if np.any(masked_corr[col_name] > CORRELATION_THRESHOLD)
    ]
    return cols_to_drop


def get_preprocessor_pipeline(
    model_name: str, dataframe: pd.DataFrame
) -> ColumnTransformer:
    """Create a preprocessing pipeline tailored to the specific model family.

    Args:
        model_name: Name of the model estimator to be selected.
        dataframe: The dataset used to infer feature schemas and correlation thresholds.

    Raises:
        ValueError: If model_name is not found.

    Returns:
        A configured Scikit-Learn ColumnTransformer instance.
    """
    numeric_features = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_features = dataframe.select_dtypes(include="string").columns.tolist()

    if model_name in ESTIMATOR_MODELS["Tree_models"]:
        return ColumnTransformer(
            transformers=[
                ("numeric", SimpleImputer(strategy="median"), numeric_features),
                (
                    "categorical",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    categorical_features,
                ),
            ],
            remainder="passthrough",
        )

    elif model_name in ESTIMATOR_MODELS["Linear_distance_models"]:
        high_correlated_num = _get_numeric_cols_to_drop(dataframe=dataframe)
        numeric_features = [
            num_feat
            for num_feat in numeric_features
            if num_feat not in high_correlated_num
        ]
        numeric_transformer = Sklearn_Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                (
                    "categorical",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
                ("rm_high_corr_cols", "drop", high_correlated_num),
            ],
            remainder="passthrough",
        )
    else:
        raise ValueError("model_name not found.")
