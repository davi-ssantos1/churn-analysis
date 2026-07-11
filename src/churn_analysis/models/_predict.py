"""Model prediction Module."""

from typing import Any

import numpy as np
import numpy.typing as npt

from churn_analysis.utils import ModelProtocol


def predict(model: ModelProtocol, X: npt.NDArray[np.float64]) -> Any:
    """Use a given model to predict the valeus for the provided features.

    Args:
        model: Estimator instance used to make predictions.
        X: Input feature array

    Returns:
        The predicted values array
    """
    y_pred = model.predict(X)
    return y_pred
