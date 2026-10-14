import numpy as np
import pandas as pd
import pytest

from credit_risk_fraud.modeling import (
    choose_threshold,
    evaluate_classifier,
    select_threshold,
    time_ordered_split,
    train_logistic_model,
)


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
    threshold = select_threshold(model, features, labels, minimum_recall=0.5)

    report = evaluate_classifier(model, features, labels, threshold)

    assert report.roc_auc > 0.9
    assert report.pr_auc > 0.9
    assert report.recall >= 0.5
    assert 0.0 <= report.precision <= 1.0
    assert report.brier < 0.1
    assert report.base_rate == pytest.approx(0.5)
    assert report.threshold == threshold


def test_time_ordered_split_keeps_later_periods_out_of_training():
    data = pd.DataFrame({"t": np.arange(100), "y": np.arange(100) % 2})

    train, validation, test = time_ordered_split(data, "t", validation_fraction=0.2, test_fraction=0.1)

    assert len(train) + len(validation) + len(test) == 100
    assert train["t"].max() < validation["t"].min()
    assert validation["t"].max() < test["t"].min()
    assert len(test) == pytest.approx(10, abs=1)


def test_time_ordered_split_keeps_tied_timestamps_together():
    data = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]})

    train, validation, test = time_ordered_split(data, "t", validation_fraction=0.3, test_fraction=0.3)

    assert set(train["t"]) == {1}
    assert set(validation["t"]) == {2}
    assert set(test["t"]) == {3}


def test_time_ordered_split_rejects_invalid_fractions():
    data = pd.DataFrame({"t": np.arange(10)})

    with pytest.raises(ValueError):
        time_ordered_split(data, "t", validation_fraction=0.6, test_fraction=0.5)
