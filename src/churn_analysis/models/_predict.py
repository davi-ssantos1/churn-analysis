"""Model prediction Module."""

from typing import Any

import numpy as np
import numpy.typing as npt
from imblearn.pipeline import Pipeline


def predict(
    model_pipeline: Pipeline, X: npt.NDArray[np.floating[Any]]
) -> npt.NDArray[np.floating[Any]]:
    """Use a given model to predict the valeus for the provided features.

    Args:
        model_pipeline: A Imblearn Pipeline instance to be used for training.
        X: Input feature matrix.

    Returns:
        The predicted values array
    """
    threshold = 0.5
    y_prob_pred: npt.NDArray[np.floating[Any]] = model_pipeline.predict_proba(X=X)[:, 1]
    y_pred_custom = (y_prob_pred >= threshold).astype(int)
    return y_pred_custom
