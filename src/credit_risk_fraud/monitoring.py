import numpy as np

# Conventional PSI reading used in credit risk model monitoring.
PSI_STABLE = 0.10
PSI_SHIFT = 0.25


def population_stability_index(expected, actual, bins: int = 10, epsilon: float = 1e-6) -> float:
    """PSI between a reference distribution (e.g. training scores) and a newer one.

    Bin edges are the reference quantiles, so each reference bin holds ~1/bins of the rows.
    PSI = sum((actual% - expected%) * ln(actual% / expected%)).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected_share = np.histogram(expected, edges)[0] / len(expected)
    actual_share = np.histogram(actual, edges)[0] / len(actual)
    expected_share = np.clip(expected_share, epsilon, None)
    actual_share = np.clip(actual_share, epsilon, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_SHIFT:
        return "moderate shift - investigate"
    return "significant shift - review model"
