from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import classification_report, roc_auc_score


def evaluate(
    y_true: npt.NDArray[np.float64],
    y_pred: npt.NDArray[np.float64],
) -> Sequence[dict]:
    report = classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
    report["roc_auc"] = roc_auc_score(y_true=y_true, y_score=y_pred)

    flat_metrics = {
        "roc_auc": report["roc_auc"],
        "accuracy": report["accuracy"],
        "precision_class_1": report["1.0"]["precision"],
        "recall_class_1": report["1.0"]["recall"],
        "f1_score_class_1": report["1.0"]["f1-score"],
        "f1_score_macro": report["macro avg"]["f1-score"],
    }

    return flat_metrics, report
