from typing import Any

import numpy as np
import numpy.typing as npt

from churn_analysis.utils import ModelProtocol


def predict(model: ModelProtocol, X: npt.NDArray[np.float64]) -> Any:
    y_pred = model.predict(X)
    return y_pred
