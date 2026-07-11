import numpy as np
import numpy.typing as npt

from churn_analysis.utils import ModelProtocol


def train(
    model: ModelProtocol,
    x_train: npt.NDArray[np.float64],
    y_train: npt.NDArray[np.float64],
) -> ModelProtocol:
    model.fit(X=x_train, y=y_train)
    return model
