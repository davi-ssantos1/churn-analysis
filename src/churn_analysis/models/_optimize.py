"""Model tuning module."""

from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from churn_analysis.config import RANDOM_STATE
from churn_analysis.feature_eng import get_preprocessor_pipeline
from churn_analysis.models._registry import get_model


def tune_hyperparemeters(
    model_name: str,
    apply_smote: bool,
    x_train: npt.NDArray[np.floating[Any]],
    y_train: npt.NDArray[np.floating[Any]],
    n_trials: int,
) -> dict[str, Any]:
    """Tune the hyperparemeters of a given model using Optuna.

    Args:
        model_name: Name of the model to be selected and tuned.
        apply_smote: Set if SMOTE oversampling will be applied in the model training step.
        x_train: Training features matrix.
        y_train: Training target labels array.
        n_trials: Number of Optuna optimization trials.

    Return:
        Dictionary containing the optimal hyperparameter mapping found by the study.
    """

    def objective(trial: optuna.Trial) -> float:
        match model_name:
            case "xgboost":
                params = {
                    "classifier__max_depth": trial.suggest_int(
                        "classifier__max_depth", low=3, high=30
                    ),
                    "classifier__learning_rate": trial.suggest_float(
                        "classifier__learning_rate", low=1e-3, high=0.1, log=True
                    ),
                    "classifier__n_estimators": trial.suggest_int(
                        "classifier__n_estimators", low=10, high=200
                    ),
                }
            case "k_neighbors":
                params = {
                    "classifier__n_neighbors": trial.suggest_int(
                        "classifier__n_neighbors", low=3, high=30
                    ),
                    "classifier__p": trial.suggest_float(
                        "classifier__p", low=1, high=2
                    ),
                }
            case "logistic_regression":
                params = {
                    "classifier__C": trial.suggest_float(
                        "classifier__C", low=1e-3, high=10, log=True
                    )
                }
            case "random_forest":
                params = {
                    "classifier__n_estimators": trial.suggest_int(
                        "classifier__n_estimators", low=50, high=200
                    ),
                    "classifier__max_depth": trial.suggest_int(
                        "classifier__max_depth", low=3, high=10
                    ),
                }
            case _:
                raise ValueError(f"{model_name} model not found.")
        if apply_smote:
            params["smote__k_neighbors"] = trial.suggest_int(
                "smote__k_neighbors", low=2, high=20
            )
        preprocessor = get_preprocessor_pipeline(
            model_name=model_name, dataframe=x_train
        )
        model = get_model(
            model_name=model_name, preprocessor=preprocessor, apply_smote=apply_smote
        )
        model.set_params(**params)

        cv = StratifiedKFold(shuffle=True, random_state=RANDOM_STATE)
        score = cross_val_score(
            estimator=model, X=x_train, y=y_train, cv=cv, scoring="roc_auc", n_jobs=5
        )
        return float(score.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(func=objective, n_trials=n_trials)
    return study.best_params
