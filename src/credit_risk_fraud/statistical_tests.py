"""Hypothesis tests for default and fraud drivers, reported with effect sizes.

With hundreds of thousands of rows almost any difference is statistically significant,
so every test also reports an effect size; the effect size, not the p-value, says
whether a driver matters in practice.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    driver: str
    outcome: str
    test: str
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    effect_magnitude: str
    detail: dict

    def as_dict(self) -> dict:
        return asdict(self)


def _magnitude(value: float, small: float, medium: float, large: float) -> str:
    value = abs(value)
    if value < small:
        return "negligible"
    if value < medium:
        return "small"
    if value < large:
        return "medium"
    return "large"


def chi_square_test(data: pd.DataFrame, driver: str, outcome: str, min_count: int = 0) -> TestResult:
    """Chi-square test of independence with Cramér's V as effect size.

    Categories with fewer than ``min_count`` rows are dropped so sparse cells do not
    invalidate the chi-square approximation.
    """
    subset = data[[driver, outcome]].dropna()
    counts = subset[driver].value_counts()
    subset = subset[subset[driver].isin(counts[counts >= min_count].index)]
    table = pd.crosstab(subset[driver], subset[outcome])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    # Cohen's thresholds scale with the smaller table dimension (df* = min(r, c) - 1).
    df_star = min(table.shape) - 1
    small, medium, large = (0.10 / np.sqrt(df_star), 0.30 / np.sqrt(df_star), 0.50 / np.sqrt(df_star))
    rates = subset.groupby(driver)[outcome].agg(["mean", "size"]).sort_values("mean", ascending=False)
    return TestResult(
        driver=driver,
        outcome=outcome,
        test="chi-square independence",
        statistic=float(chi2),
        p_value=float(p_value),
        effect_size_name="Cramer's V",
        effect_size=cramers_v,
        effect_magnitude=_magnitude(cramers_v, small, medium, large),
        detail={
            "degrees_of_freedom": int(dof),
            "rows": int(n),
            "highest_rate": {str(k): round(float(v), 4) for k, v in rates["mean"].head(3).items()},
            "lowest_rate": {str(k): round(float(v), 4) for k, v in rates["mean"].tail(3).items()},
        },
    )


def mann_whitney_test(data: pd.DataFrame, driver: str, outcome: str) -> TestResult:
    """Mann-Whitney U test of a numeric driver between outcome groups.

    Rank-biserial correlation is the effect size: the probability a positive-class row has
    a higher value than a negative-class row, rescaled to [-1, 1].
    """
    subset = data[[driver, outcome]].dropna()
    positives = subset.loc[subset[outcome] == 1, driver].to_numpy()
    negatives = subset.loc[subset[outcome] == 0, driver].to_numpy()
    statistic, p_value = stats.mannwhitneyu(positives, negatives, alternative="two-sided")
    rank_biserial = float(2 * statistic / (len(positives) * len(negatives)) - 1)
    return TestResult(
        driver=driver,
        outcome=outcome,
        test="Mann-Whitney U",
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size_name="rank-biserial r",
        effect_size=rank_biserial,
        effect_magnitude=_magnitude(rank_biserial, 0.1, 0.3, 0.5),
        detail={
            "rows": len(subset),
            "median_positive": float(np.median(positives)),
            "median_negative": float(np.median(negatives)),
        },
    )


def bonferroni_alpha(results: list[TestResult], alpha: float = 0.05) -> float:
    return alpha / max(len(results), 1)
