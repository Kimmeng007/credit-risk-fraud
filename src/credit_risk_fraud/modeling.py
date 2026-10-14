from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    brier: float
    base_rate: float
    threshold: float
    precision: float
    recall: float


def time_ordered_split(data, time_column: str, validation_fraction: float = 0.15,
                       test_fraction: float = 0.15):
    """Split rows into train/validation/test by time so later periods are never seen in training.

    Rows sharing the boundary timestamp stay in the same split, so the fractions are approximate.
    """
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must be positive and sum to less than 1")
    times = data[time_column]
    validation_start = times.quantile(1 - validation_fraction - test_fraction, interpolation="lower")
    test_start = times.quantile(1 - test_fraction, interpolation="lower")
    if validation_start == test_start:
        raise ValueError(f"{time_column} has too few distinct values for a three-way time split")
    train = data[times < validation_start]
    validation = data[(times >= validation_start) & (times < test_start)]
    test = data[times >= test_start]
    return train, validation, test


def choose_threshold(y_true, probabilities, minimum_recall: float = 0.50) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if len(eligible) == 0:
        return 0.5
    best = eligible[np.argmax(precision[:-1][eligible])]
    return float(thresholds[best])


def train_logistic_model(features, labels, class_weight: str = "balanced"):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42),
    )
    model.fit(features, labels)
    return model


def select_threshold(model, features, labels, minimum_recall: float = 0.50) -> float:
    """Pick the decision threshold on a validation set, never on the test set it is reported on."""
    return choose_threshold(labels, model.predict_proba(features)[:, 1], minimum_recall)


def evaluate_classifier(model, features, labels, threshold: float) -> ClassificationReport:
    labels = np.asarray(labels)
    probabilities = model.predict_proba(features)[:, 1]
    predictions = probabilities >= threshold
    true_positives = np.sum(predictions & (labels == 1))
    predicted_positives = np.sum(predictions)
    actual_positives = np.sum(labels == 1)
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    return ClassificationReport(
        roc_auc=float(roc_auc_score(labels, probabilities)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        # Class weighting inflates probabilities, so Brier here measures ranking-model
        # calibration as trained; recalibrate before using scores as PDs.
        brier=float(brier_score_loss(labels, probabilities)),
        base_rate=float(labels.mean()),
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
    )
