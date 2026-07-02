"""Statistical significance testing: is a change between two groups a real signal,
or could it just be random variation? Complements agent.drift's "what changed" and
"which segment drove it" with a third question drift alone can't answer — pure
scipy.stats, zero AI cost, and — like everything else in this app — the AI is only
ever allowed to narrate these pre-computed, verified numbers, never invent its own.
"""

import math

import pandas as pd
from scipy import stats

DEFAULT_ALPHA = 0.05


def compare_numeric_significance(before: pd.Series, after: pd.Series, alpha: float = DEFAULT_ALPHA) -> dict:
    """Welch's t-test (does not assume equal variance, the safer default for two
    real-world samples) on whether two numeric samples' means differ significantly."""
    before = before.dropna()
    after = after.dropna()
    if len(before) < 2 or len(after) < 2:
        return {
            "test": "welch_t_test",
            "applicable": False,
            "reason": "Need at least 2 non-missing values in each group.",
        }
    result = stats.ttest_ind(before, after, equal_var=False)
    p_value = float(result.pvalue)
    return {
        "test": "welch_t_test",
        "applicable": True,
        "statistic": float(result.statistic),
        "p_value": p_value,
        "alpha": alpha,
        "significant": p_value < alpha,
        "n_before": int(len(before)),
        "n_after": int(len(after)),
    }


def compare_categorical_significance(before: pd.Series, after: pd.Series, alpha: float = DEFAULT_ALPHA) -> dict:
    """Chi-square test of homogeneity: do the two groups' category proportions
    differ significantly? Values are stringified first so columns with unhashable
    cell values (dicts/lists from nested JSON) can't crash the set operations below."""
    before_counts = {str(k): int(v) for k, v in before.dropna().value_counts().items()}
    after_counts = {str(k): int(v) for k, v in after.dropna().value_counts().items()}
    categories = sorted(set(before_counts) | set(after_counts))
    if len(categories) < 2:
        return {
            "test": "chi_square",
            "applicable": False,
            "reason": "Need at least 2 distinct categories across both groups.",
        }
    table = [
        [before_counts.get(c, 0) for c in categories],
        [after_counts.get(c, 0) for c in categories],
    ]
    try:
        chi2, p_value, dof, _expected = stats.chi2_contingency(table)
    except ValueError as exc:
        return {"test": "chi_square", "applicable": False, "reason": str(exc)}
    p_value = float(p_value)
    return {
        "test": "chi_square",
        "applicable": True,
        "statistic": float(chi2),
        "p_value": p_value,
        "dof": int(dof),
        "alpha": alpha,
        "significant": p_value < alpha,
    }


def compare_conversion_rates(
    count_a: int, total_a: int, count_b: int, total_b: int, alpha: float = DEFAULT_ALPHA
) -> dict:
    """Two-proportion z-test — the standard A/B test for a single binary outcome
    rate (conversion, click-through, churn). Distinct from
    compare_categorical_significance, which tests whether an entire categorical
    distribution shifted; this tests one specific rate, with a confidence
    interval on the difference rather than just significant/not-significant."""
    test = "two_proportion_z_test"
    if total_a <= 0 or total_b <= 0:
        return {"test": test, "applicable": False, "reason": "Each group needs at least 1 observation."}
    if not (0 <= count_a <= total_a) or not (0 <= count_b <= total_b):
        return {"test": test, "applicable": False, "reason": "count must be between 0 and total for each group."}

    rate_a = count_a / total_a
    rate_b = count_b / total_b
    pooled_rate = (count_a + count_b) / (total_a + total_b)

    se_pooled = math.sqrt(pooled_rate * (1 - pooled_rate) * (1 / total_a + 1 / total_b))
    if se_pooled == 0:
        return {
            "test": test,
            "applicable": False,
            "reason": "No variance in the pooled rate (e.g. both groups at 0% or 100%).",
        }

    z = (rate_b - rate_a) / se_pooled
    p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

    se_unpooled = math.sqrt(rate_a * (1 - rate_a) / total_a + rate_b * (1 - rate_b) / total_b)
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    diff = rate_b - rate_a

    return {
        "test": test,
        "applicable": True,
        "rate_a": round(rate_a, 4),
        "rate_b": round(rate_b, 4),
        "count_a": count_a,
        "total_a": total_a,
        "count_b": count_b,
        "total_b": total_b,
        "diff": round(diff, 4),
        "ci_low": round(diff - z_crit * se_unpooled, 4),
        "ci_high": round(diff + z_crit * se_unpooled, 4),
        "alpha": alpha,
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < alpha,
    }
