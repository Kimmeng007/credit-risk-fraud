from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    threshold: float
    precision: float
    recall: float


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


def evaluate_classifier(model, features, labels, minimum_recall: float = 0.50) -> ClassificationReport:
    probabilities = model.predict_proba(features)[:, 1]
    threshold = choose_threshold(labels, probabilities, minimum_recall)
    predictions = probabilities >= threshold
    true_positives = np.sum((predictions == 1) & (labels == 1))
    predicted_positives = np.sum(predictions == 1)
    actual_positives = np.sum(labels == 1)
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    return ClassificationReport(
        roc_auc=float(roc_auc_score(labels, probabilities)),
        pr_auc=float(average_precision_score(labels, probabilities)),
        threshold=threshold,
        precision=float(precision),
        recall=float(recall),
    )