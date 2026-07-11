import json
import os
import tempfile

import mlflow
import numpy as np
import numpy.typing as npt

from churn_analysis.config import MODEL_NAMES
from churn_analysis.models._evaluate import evaluate
from churn_analysis.models._optimize import tune_hyperparemeters
from churn_analysis.models._predict import predict
from churn_analysis.models._registry import get_model
from churn_analysis.models._train import train


def _mlflow_log_artifact_json(report: dict, model_name: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(
            temp_dir, f"{model_name}_classification_report.json"
        )

        with open(temp_file_path, "w") as file:
            json.dump(obj=report, fp=file, indent=4)
        mlflow.log_artifact(local_path=temp_file_path)


def _get_winner_model(
    x_train: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
) -> str:
    mlflow.set_experiment(experiment_name="churn_analys_get_winner_model")
    best_model_name = " "
    best_roc_auc = 0
    for model_name in MODEL_NAMES:
        with mlflow.start_run(run_name=f"{model_name}_baseline"):
            model = get_model(model_name=model_name)
            model = train(model=model, x_train=x_train, y_train=y_train)
            mlflow.log_params(params=model.get_params())
            y_pred = predict(model=model, X=x_test)
            flat_metrics, report = evaluate(y_true=y_test, y_pred=y_pred)
            mlflow.log_metrics(metrics=flat_metrics)
            _mlflow_log_artifact_json(report=report, model_name=model_name)
            if flat_metrics["roc_auc"] > best_roc_auc:
                best_model_name = model_name
                best_roc_auc = flat_metrics["roc_auc"]

    return best_model_name


def models_pipeline(
    x_train: npt.NDArray[np.float64],
    x_test: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
    y_test: npt.NDArray[np.float64],
) -> str:
    best_model_name = _get_winner_model(
        x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test
    )
    best_model_params = tune_hyperparemeters(
        model_name=best_model_name,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    return best_model_params
