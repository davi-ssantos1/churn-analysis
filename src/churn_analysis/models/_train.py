"""Model pipeline training engine module."""

from typing import Any

import numpy as np
import numpy.typing as npt
from imblearn.pipeline import Pipeline


def train(
    model_pipeline: Pipeline,
    x_train: npt.NDArray[np.floating[Any]],
    y_train: npt.NDArray[np.floating[Any]],
) -> Pipeline:
    """Execute the traing phase for a given model pipeline.

    Args:
        model_pipeline: A Imblearn Pipeline instance to be used for training.
        x_train: Traning input features matrix.
        y_train: Traning target labels array.

    Returns:
        The fitted model instance
    """
    model_pipeline = model_pipeline.fit(X=x_train, y=y_train)
    return model_pipeline
