import numpy as np
import pytest

from credit_risk_fraud.modeling import choose_threshold, evaluate_classifier, train_logistic_model


@pytest.fixture
def separable_data():
    rng = np.random.default_rng(42)
    n = 400
    negatives = rng.normal(loc=-2.0, scale=1.0, size=(n // 2, 2))
    positives = rng.normal(loc=2.0, scale=1.0, size=(n // 2, 2))
    features = np.vstack([negatives, positives])
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    return features, labels


def test_choose_threshold_meets_minimum_recall():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.9])

    threshold = choose_threshold(y_true, probabilities, minimum_recall=0.66)

    predicted_positive = probabilities >= threshold
    recall = np.sum(predicted_positive & (y_true == 1)) / np.sum(y_true == 1)
    assert recall >= 0.66


def test_choose_threshold_falls_back_when_unreachable():
    # Recall of 1.0 is always reachable by predicting everything positive, so
    # only an impossible target (>1.0) can leave no eligible threshold.
    y_true = np.array([0, 1])
    probabilities = np.array([0.1, 0.2])

    assert choose_threshold(y_true, probabilities, minimum_recall=1.01) == 0.5


def test_train_logistic_model_separates_clusters(separable_data):
    features, labels = separable_data

    model = train_logistic_model(features, labels)
    predictions = model.predict(features)

    assert (predictions == labels).mean() > 0.9


def test_evaluate_classifier_reports_expected_fields(separable_data):
    features, labels = separable_data
    model = train_logistic_model(features, labels)

    report = evaluate_classifier(model, features, labels, minimum_recall=0.5)

    assert report.roc_auc > 0.9
    assert report.pr_auc > 0.9
    assert report.recall >= 0.5
    assert 0.0 <= report.precision <= 1.0
