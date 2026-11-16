"""Statistics used throughout: data splits, AUROC, effect size, bootstrap.

Everything here is plain NumPy and works on small arrays, so the code favours
readability over speed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def split_groups_in_half(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly assign each distinct group (here: each question) to half A or B.

    Returns a boolean array with one entry per item, True for items whose group
    landed in half A. All items of a question stay together, so the two halves
    share no questions.
    """
    unique = np.unique(groups)
    in_a = rng.permutation(unique)[: len(unique) // 2]
    return np.isin(groups, in_a)


# --------------------------------------------------------------------------- #
# Separation measures
# --------------------------------------------------------------------------- #


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Area under the ROC curve for telling positives from negatives by score.

    It equals the probability that a randomly chosen positive scores higher
    than a randomly chosen negative (ties count as half):

        AUROC = P(pos > neg) + 0.5 * P(pos == neg)

    1.0 means every positive is above every negative, 0.5 means the score
    carries no information, and 0.0 means perfectly separated but with the
    sign reversed.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    diff = pos[:, None] - neg[None, :]  # every positive against every negative
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def cohens_d(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Difference of the two means in units of their pooled standard deviation.

        d = (mean(pos) - mean(neg)) / sqrt((var(pos) + var(neg)) / 2)

    Unlike AUROC it does not saturate: two sets that are already perfectly
    separated (AUROC = 1) can still differ in how far apart they are.
    """
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2)
    return float((pos.mean() - neg.mean()) / (pooled + 1e-12))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: 1 same direction, 0 orthogonal."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_over_groups(
    statistic: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for ``statistic`` by resampling whole groups.

    Responses to the same question are not independent, so the unit we
    resample is the question, not the response. One bootstrap replicate:

    1. draw as many questions as there are, *with replacement*;
    2. collect the indices of all items belonging to the drawn questions (a
       question drawn twice contributes its items twice);
    3. compute ``statistic(indices)``.

    The interval is the central ``level`` fraction of the replicate values
    (the "percentile bootstrap"). With few questions the interval is coarse:
    it can only reflect the variation among the questions we happen to have.

    ``statistic`` receives an integer index array and returns a number. It may
    return NaN for a degenerate resample; NaNs are ignored.
    """
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}

    values = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[g] for g in drawn])
        values[b] = statistic(indices)

    tail = (1 - level) / 2
    low, high = np.nanquantile(values, [tail, 1 - tail])
    return float(low), float(high)


def mean_with_ci(
    values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """Mean of ``values`` with a bootstrap interval over groups: ``(mean, low, high)``."""
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_over_groups(lambda idx: values[idx].mean(), groups, n_boot, seed)
    return float(values.mean()), low, high


# --------------------------------------------------------------------------- #
# Rank correlation
# --------------------------------------------------------------------------- #


def ranks(values: np.ndarray) -> np.ndarray:
    """Rank of each value, 1 for the smallest. Tied values share the average
    of the ranks they cover, so ``[10, 20, 20, 30]`` gets ``[1, 2.5, 2.5, 4]``."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
            end += 1
        result[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: the ordinary (Pearson) correlation of the
    ranks of ``x`` and ``y``.

    It asks whether the two orderings agree, not whether the numbers match:
    a judge that always scores 20 points too high still gets 1.0. Returns NaN
    when either input is constant, because a constant has no ordering.
    """
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_with_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, groups: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Spearman correlation with a percentile bootstrap interval.

    Items are resampled one by one, or in whole groups if ``groups`` is given.
    Returns ``(correlation, low, high)``.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    groups = np.arange(len(x)) if groups is None else np.asarray(groups)
    low, high = bootstrap_over_groups(lambda idx: spearman(x[idx], y[idx]), groups, n_boot, seed)
    return spearman(x, y), low, high
