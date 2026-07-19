"""Model pipeline orchestrator and tracker module."""

import json
import os
import tempfile
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from churn_analysis.config import ESTIMATOR_MODELS
from churn_analysis.feature_eng import get_preprocessor_pipeline
from churn_analysis.models._evaluate import evaluate
from churn_analysis.models._optimize import tune_hyperparemeters
from churn_analysis.models._predict import predict
from churn_analysis.models._registry import get_model
from churn_analysis.models._train import train

OPTUNA_N_TRIALS_TOURNMENT = 10
OPTUNA_N_TRAILS_WINNER_MODEL = 20


class ModelsPipeline:
    """Orchestrates end-to-end model selection, hyperparameter tuning, and MLflow logging."""

    def __init__(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """Initialize the pipeline with training and testing datasets.

        Args:
            x_train: Training features matrix.
            x_test: Testing feature matrix.
            y_train: Training target labels array.
            y_test: Testing target labels array.
        """
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test

    def run(self) -> tuple[str, dict[str, Any]]:
        """Execute the model orchestrastion and tracking pipeline.

        Returns:
            Tuple of best model name and a dictionary containing the evaluation metrics of the best performing model.
        """
        best_model_name, apply_smote = self._get_winner_model()
        best_model_params = tune_hyperparemeters(
            model_name=best_model_name,
            apply_smote=apply_smote,
            x_train=self._x_train,
            y_train=self._y_train,
            n_trials=OPTUNA_N_TRAILS_WINNER_MODEL,
        )

        mlflow.set_experiment(experiment_name="Churn_analysis_Production_model")
        run_name = (
            f"{best_model_name}_tuned_training_SMOTE"
            if apply_smote
            else f"{best_model_name}_tuned_training"
        )
        with mlflow.start_run(run_name=run_name):
            preprocessor = get_preprocessor_pipeline(
                model_name=best_model_name, dataframe=self._x_train
            )
            model_pipeline = get_model(
                model_name=best_model_name,
                preprocessor=preprocessor,
                apply_smote=apply_smote,
            )
            model_pipeline.set_params(**best_model_params)
            model_pipeline = train(
                model_pipeline=model_pipeline,
                x_train=self._x_train,
                y_train=self._y_train,
            )
            input_sample = self._x_train.head(5)
            output_sample = predict(model_pipeline=model_pipeline, X=self._x_train)
            signature = infer_signature(
                model_input=input_sample, model_output=output_sample
            )
            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                name="model",
                registered_model_name=best_model_name,
                signature=signature,
                input_example=input_sample,
                skops_trusted_types=[
                    "imblearn.pipeline.Pipeline",
                    "imblearn.over_sampling._smote.base.SMOTE",
                    "numpy.dtype",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )
            y_pred = predict(model_pipeline=model_pipeline, X=self._x_test)
            flat_metrics, report = evaluate(y_true=self._y_test, y_pred=y_pred)
            self._mlflow_log_artifact_json(
                metrics=flat_metrics,
                model_name=best_model_name,
                full_report=False,
                smote=apply_smote,
            )
            self._mlflow_log_artifact_json(
                metrics=report,
                model_name=best_model_name,
                full_report=True,
                smote=apply_smote,
            )

        return best_model_name, flat_metrics

    def _get_winner_model(self) -> tuple[str, bool]:
        mlflow.set_experiment(experiment_name="Churn_analysis_Get_winner_model")
        best_model_name = " "
        apply_smote: bool
        best_roc_auc = 0
        model_names = [
            *ESTIMATOR_MODELS["Tree_models"],
            *ESTIMATOR_MODELS["Linear_distance_models"],
        ]
        for model_name in model_names:
            preprocessor = get_preprocessor_pipeline(
                model_name=model_name, dataframe=self._x_train
            )
            for smote in [True, False]:
                run_name = (
                    f"{model_name}_baseline_SMOTE"
                    if smote
                    else f"{model_name}_baseline"
                )
                with mlflow.start_run(run_name=run_name):
                    model_pipeline = get_model(
                        model_name=model_name,
                        preprocessor=preprocessor,
                        apply_smote=smote,
                    )
                    model_params = tune_hyperparemeters(
                        model_name=model_name,
                        apply_smote=smote,
                        x_train=self._x_train,
                        y_train=self._y_train,
                        n_trials=OPTUNA_N_TRIALS_TOURNMENT,
                    )
                    model_pipeline.set_params(**model_params)
                    model_pipeline = train(
                        model_pipeline=model_pipeline,
                        x_train=self._x_train,
                        y_train=self._y_train,
                    )
                    mlflow.log_params(params=model_pipeline.get_params())
                    y_pred = predict(model_pipeline=model_pipeline, X=self._x_test)
                    flat_metrics, report = evaluate(y_true=self._y_test, y_pred=y_pred)
                    mlflow.log_metrics(metrics=flat_metrics)
                    self._mlflow_log_artifact_json(
                        metrics=report,
                        model_name=model_name,
                        full_report=True,
                        smote=smote,
                    )
                    if flat_metrics["roc_auc"] > best_roc_auc:
                        best_model_name = model_name
                        apply_smote = smote
                        best_roc_auc = flat_metrics["roc_auc"]
        return best_model_name, apply_smote

    def _mlflow_log_artifact_json(
        self, metrics: dict[str, Any], model_name: str, full_report: bool, smote: bool
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            smote_str = "_SMOTE" if smote else ""
            file_name = (
                f"{model_name}_classification_full_report{smote_str}.json"
                if full_report
                else f"{model_name}_classification_flat_metrics{smote_str}.json"
            )
            temp_file_path = os.path.join(temp_dir, file_name)

            with open(temp_file_path, "w") as file:
                json.dump(obj=metrics, fp=file, indent=4)
            mlflow.log_artifact(local_path=temp_file_path)
