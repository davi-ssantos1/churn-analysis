"""Model training engine module."""

import numpy as np
import numpy.typing as npt

from churn_analysis.utils import ModelProtocol


def train(
    model: ModelProtocol,
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
) -> ModelProtocol:
    """Execute the traing phase for a given module.

    Args:
        model: Estimator instance to be trained
        x_train: Traning input features array
        y_train: Traning target labels array

    Returns:
        The fitted model instance
    """
    model.fit(X=x_train, y=y_train)
    return model
