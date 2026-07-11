import numpy as np
import numpy.typing as npt
import optuna

from churn_analysis.models._evaluate import evaluate
from churn_analysis.models._predict import predict
from churn_analysis.models._registry import get_model
from churn_analysis.models._train import train


def tune_hyperparemeters(
    model_name: str,
    x_train: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
) -> None:

    def objective(trial: optuna.Trial) -> float:
        match model_name:
            case "xgboost":
                params = {
                    "max_depth": trial.suggest_int("max_depth", low=3, high=9),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", low=1e-3, high=0.1, log=True
                    ),
                }
            case "k_nearest_neighbors":
                params = {
                    "n_neighbors": trial.suggest_int("n_neighbors", low=3, high=30),
                    "p": trial.suggest_float("p", low=1, high=2),
                }
            case "logistic_regression":
                params = {"C": trial.suggest_float("C", low=1e-3, high=10, log=True)}
            case "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", low=50, high=200),
                    "max_depth": trial.suggest_int("max_depth", low=3, high=10),
                }
            case _:
                raise ValueError(f"{model_name} model not found.")
        model = get_model(model_name=model_name)
        model.set_params(**params)
        model = train(model=model, x_train=x_train, y_train=y_train)
        y_pred = predict(model=model, X=x_test)
        flat_metrics, _ = evaluate(y_true=y_test, y_pred=y_pred)
        return flat_metrics["recall_class_1"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(func=objective, n_trials=20)
    return study.best_params
