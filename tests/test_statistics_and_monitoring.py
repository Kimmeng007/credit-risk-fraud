import numpy as np
import pandas as pd
import pytest

from credit_risk_fraud.monitoring import population_stability_index, psi_status
from credit_risk_fraud.statistical_tests import bonferroni_alpha, chi_square_test, mann_whitney_test


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_psi_is_near_zero_for_same_distribution(rng):
    reference = rng.normal(size=50_000)
    current = rng.normal(size=50_000)

    psi = population_stability_index(reference, current)

    assert psi < 0.01
    assert psi_status(psi) == "stable"


def test_psi_flags_a_shifted_distribution(rng):
    reference = rng.normal(size=50_000)
    shifted = rng.normal(loc=1.0, size=50_000)

    psi = population_stability_index(reference, shifted)

    assert psi > 0.25
    assert psi_status(psi) == "significant shift - review model"


def test_psi_ignores_missing_values_and_constant_reference():
    assert population_stability_index([1.0, np.nan, 1.0], [1.0, 1.0]) == 0.0


def test_chi_square_detects_dependence_and_reports_cramers_v(rng):
    group = rng.choice(["a", "b"], size=20_000)
    outcome = (rng.random(20_000) < np.where(group == "a", 0.10, 0.40)).astype(int)
    data = pd.DataFrame({"group": group, "outcome": outcome})

    result = chi_square_test(data, "group", "outcome")

    assert result.p_value < 1e-10
    assert 0.2 < result.effect_size < 0.5
    assert result.effect_magnitude in {"small", "medium"}
    assert result.detail["highest_rate"].keys() >= {"b"}


def test_chi_square_drops_sparse_categories(rng):
    data = pd.DataFrame({
        "group": ["a"] * 500 + ["b"] * 500 + ["rare"] * 5,
        "outcome": rng.integers(0, 2, size=1005),
    })

    result = chi_square_test(data, "group", "outcome", min_count=100)

    assert result.detail["rows"] == 1000


def test_mann_whitney_effect_size_sign_follows_positive_class(rng):
    data = pd.DataFrame({
        "value": np.concatenate([rng.normal(1.0, 1, 2000), rng.normal(0.0, 1, 2000)]),
        "outcome": [1] * 2000 + [0] * 2000,
    })

    result = mann_whitney_test(data, "value", "outcome")

    assert result.effect_size > 0.3  # positives tend to be larger
    assert result.detail["median_positive"] > result.detail["median_negative"]


def test_bonferroni_alpha_divides_by_number_of_tests(rng):
    data = pd.DataFrame({"g": rng.choice(["a", "b"], 100), "y": rng.integers(0, 2, 100)})
    results = [chi_square_test(data, "g", "y")] * 4

    assert bonferroni_alpha(results) == pytest.approx(0.0125)
