"""G-test and Holm analysis of the fixed statistical plan, separate from any sampler."""
import math


def g_test(counts, probabilities):
    from scipy.stats import chi2
    if len(counts) != len(probabilities) or sum(counts) <= 0:
        raise ValueError("nonempty matching counts/probabilities required")
    if any(n < 0 or int(n) != n for n in counts):
        raise ValueError("counts must be nonnegative integers")
    if any(p < 0 for p in probabilities) or not math.isclose(sum(probabilities), 1):
        raise ValueError("probabilities must sum to one")
    if any(n and not p for n, p in zip(counts, probabilities)):
        return {"g": None, "p_value": 0.0, "hard_failure": "observed forbidden category"}
    size = sum(counts)
    positive = sum(p > 0 for p in probabilities)
    statistic = max(0., 2 * math.fsum(n * math.log(n / (size * p))
                                     for n, p in zip(counts, probabilities) if n))
    p_value = 1.0 if positive == 1 else float(chi2.sf(statistic, positive - 1))
    return {"g": statistic, "p_value": p_value, "hard_failure": None}


def holm(p_values, alpha=.01):
    rejected = [False] * len(p_values)
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        if p_values[index] <= alpha / (len(p_values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def power_bound(plan):
    from scipy.stats import chi2
    n, delta, m = plan['samples_per_cell'], plan['target_total_variation'], plan['max_categories']
    threshold = float(chi2.isf(plan['family_error_rate'] / plan['family_cells'], m - 1))
    sufficient_g = n * delta * delta
    failure_bound = min(1., 2 ** m * math.exp(-n * delta * delta / 2))
    return {"minimum_alternative_g": sufficient_g, "worst_critical_g": threshold,
            "power_lower_bound": 1 - failure_bound if sufficient_g > threshold else 0}
