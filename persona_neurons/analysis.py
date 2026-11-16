"""Per-layer tables computed from saved activations: the persona vector,
separation and split-half reliability.

Everything here works on NumPy arrays that are already on disk, so none of it
needs the model or a GPU.

Shapes used throughout:

* ``acts``    ``[responses, layers, hidden]``  mean response activation
* ``is_pos``  ``[responses]`` booleans, True for trait-eliciting responses
* ``groups``  ``[responses]`` the question each response answers
"""

from __future__ import annotations

import re

import numpy as np

from persona_neurons.baselines import random_unit_directions, shuffled_label_vectors
from persona_neurons.persona import difference_of_means, project, unit
from persona_neurons.stats import (
    auroc,
    bootstrap_over_groups,
    cohens_d,
    cosine,
    spearman_with_ci,
    split_groups_in_half,
)


def persona_vectors(acts: np.ndarray, norms: np.ndarray, is_pos: np.ndarray) -> dict:
    """The persona vector at every layer and the numbers needed to use it.

    Returns a dict of arrays:

    * ``vectors``             ``[layers, hidden]`` mean eliciting minus mean suppressing
    * ``unit_vectors``        the same, scaled to length 1 (``v_hat``)
    * ``vector_norm``         ``[layers]`` length of each raw vector, ``||v||``
    * ``mean_residual_norm``  ``[layers]`` mean length of the residual stream
      over response tokens, used to scale steering
    """
    vectors = difference_of_means(acts, is_pos)
    return {
        "vectors": vectors,
        "unit_vectors": unit(vectors),
        "vector_norm": np.linalg.norm(vectors, axis=-1),
        "mean_residual_norm": norms.mean(axis=0),
    }


def separation_table(
    vectors: dict,
    acts_extract: np.ndarray,
    is_pos_extract: np.ndarray,
    groups_extract: np.ndarray,
    acts_controlled: np.ndarray,
    acts_own: np.ndarray,
    is_pos_test: np.ndarray,
    groups_test: np.ndarray,
    n_bootstrap: int,
    n_baseline: int,
    seed: int,
) -> list[dict]:
    """How well does the projection on the persona vector tell trait-eliciting
    from trait-suppressing responses, at each layer?

    Two encodings of the same held-out responses are scored:

    * ``acts_controlled``: re-encoded under one neutral system prompt. The
      prompt is identical for both sets, so only the *content of the response*
      can separate them. This is the primary test.
    * ``acts_own``: encoded under the system prompts that generated them. The
      prompt alone shifts response-token activations, so this test can pass
      trivially. Reported for comparison.

    Two baselines run through the same prompt-controlled computation:

    * shuffled-label vectors (labels shuffled within question on the
      extraction data);
    * uniformly random unit directions.

    Returns one dict per layer. AUROCs have a percentile bootstrap interval
    over questions.
    """
    directions = vectors["unit_vectors"]
    n_layers = directions.shape[0]
    proj_controlled = project(acts_controlled, directions)  # [responses, layers]
    proj_own = project(acts_own, directions)
    shuffled_dirs = unit(
        shuffled_label_vectors(acts_extract, is_pos_extract, groups_extract, n_baseline, seed=seed)
    )  # [shuffles, layers, hidden]
    random_dirs = random_unit_directions(n_baseline, directions.shape[1], seed=seed)

    def baseline_aurocs(candidate_dirs: np.ndarray, layer: int) -> np.ndarray:
        """AUROC of each candidate direction on the prompt-controlled activations."""
        scores = acts_controlled[:, layer] @ candidate_dirs.T  # [responses, candidates]
        return np.array(
            [auroc(scores[is_pos_test, c], scores[~is_pos_test, c]) for c in range(scores.shape[1])]
        )

    def auroc_with_ci(scores: np.ndarray) -> tuple[float, float, float]:
        def stat(idx):
            p, n = scores[idx][is_pos_test[idx]], scores[idx][~is_pos_test[idx]]
            return auroc(p, n) if len(p) and len(n) else np.nan

        low, high = bootstrap_over_groups(stat, groups_test, n_bootstrap, seed=seed)
        return auroc(scores[is_pos_test], scores[~is_pos_test]), low, high

    rows = []
    for layer in range(n_layers):
        controlled = auroc_with_ci(proj_controlled[:, layer])
        own = auroc_with_ci(proj_own[:, layer])
        shuffled_aurocs = baseline_aurocs(shuffled_dirs[:, layer], layer)
        folded = np.maximum(shuffled_aurocs, 1 - shuffled_aurocs)  # sign-free version
        random_aurocs = baseline_aurocs(random_dirs, layer)
        random_folded = np.maximum(random_aurocs, 1 - random_aurocs)
        # A shuffled vector is a random mix of the same activations, so it can
        # align with the real vector by chance. These two numbers show how much
        # it does, and whether that alignment explains its AUROC.
        shuffled_cos = shuffled_dirs[:, layer] @ directions[layer]
        rows.append(
            {
                "layer": layer,
                "auroc_controlled": controlled[0],
                "auroc_controlled_low": controlled[1],
                "auroc_controlled_high": controlled[2],
                "auroc_own": own[0],
                "auroc_own_low": own[1],
                "auroc_own_high": own[2],
                "cohens_d_controlled": cohens_d(
                    proj_controlled[is_pos_test, layer], proj_controlled[~is_pos_test, layer]
                ),
                "shuffled_mean": float(shuffled_aurocs.mean()),
                "shuffled_p2_5": float(np.quantile(shuffled_aurocs, 0.025)),
                "shuffled_p97_5": float(np.quantile(shuffled_aurocs, 0.975)),
                "shuffled_folded_p95": float(np.quantile(folded, 0.95)),
                "shuffled_folded_max": float(folded.max()),
                "shuffled_abs_cosine_with_vector": float(np.abs(shuffled_cos).mean()),
                "shuffled_cosine_auroc_correlation": float(
                    np.corrcoef(shuffled_cos, shuffled_aurocs)[0, 1]
                ),
                "random_p2_5": float(np.quantile(random_aurocs, 0.025)),
                "random_p97_5": float(np.quantile(random_aurocs, 0.975)),
                "random_folded_p95": float(np.quantile(random_folded, 0.95)),
                "vector_norm": float(vectors["vector_norm"][layer]),
                "mean_residual_norm": float(vectors["mean_residual_norm"][layer]),
            }
        )
    return rows


def reliability_table(
    acts: np.ndarray, is_pos: np.ndarray, groups: np.ndarray, seed: int, n_random_splits: int
) -> tuple[dict, list[dict]]:
    """Split-half reliability: do two disjoint halves of the questions give the
    same vector?

    The questions are divided at random into halves A and B, a vector is
    extracted from each, and the cosine similarity between the two is reported
    per layer. A value near 1 means the direction does not depend on which
    questions were used. One fixed split is the headline number; further
    random splits show how much that number depends on the split.

    Returns ``(halves, rows)``: the question ids of the fixed split, and one
    dict per layer.
    """
    n_layers = acts.shape[1]

    def half_cosines(in_a: np.ndarray) -> np.ndarray:
        """Cosine per layer between vectors from two disjoint sets of questions."""
        v_a = difference_of_means(acts[in_a], is_pos[in_a])
        v_b = difference_of_means(acts[~in_a], is_pos[~in_a])
        return np.array([cosine(v_a[layer], v_b[layer]) for layer in range(n_layers)])

    rng = np.random.default_rng(seed)
    in_a_fixed = split_groups_in_half(groups, rng)
    fixed = half_cosines(in_a_fixed)
    random_splits = np.stack(
        [half_cosines(split_groups_in_half(groups, rng)) for _ in range(n_random_splits)]
    )
    rows = [
        {
            "layer": layer,
            "cosine": float(fixed[layer]),
            "cosine_random_mean": float(random_splits[:, layer].mean()),
            "cosine_random_min": float(random_splits[:, layer].min()),
            "cosine_random_max": float(random_splits[:, layer].max()),
        }
        for layer in range(n_layers)
    ]
    halves = {
        "half_a_questions": sorted(set(groups[in_a_fixed].tolist())),
        "half_b_questions": sorted(set(groups[~in_a_fixed].tolist())),
    }
    return halves, rows


def numbered_list_fraction(records: list[dict], condition: str) -> float:
    """Share of the responses in one condition that contain a numbered list.

    The two conditions can differ in format as well as in content, and a
    difference-of-means vector picks up both, so we record how large the
    format difference is.
    """
    chosen = [r["response"] for r in records if r["condition"] == condition]
    return float(np.mean([bool(re.search(r"(^|\n)\s*1\.", text)) for text in chosen]))


def filter_pairs(
    records: list[dict],
    scores: dict[str, dict],
    trait_threshold: float,
    coherence_threshold: float,
) -> tuple[set[str], dict]:
    """Keep only response pairs in which the persona prompts worked.

    Following Chen et al. (``generate_vec.py``), responses are paired: the
    trait-eliciting and trait-suppressing responses to the same question,
    instruction pair and sample number. A pair is kept when

    * the eliciting response scores at least ``trait_threshold`` for the trait,
    * the suppressing response scores below ``100 - trait_threshold``, and
    * both score at least ``coherence_threshold`` for coherence.

    A response the judge could not score counts as failing. Filtering matters
    because a difference of means is diluted by every "positive" response in
    which the model ignored the prompt.

    Returns the ids of the kept records and a dict of counts for the log.
    """
    by_pair: dict[tuple, dict[str, dict]] = {}
    for r in records:
        key = (r["question_idx"], r["instruction_idx"], r["sample_idx"])
        by_pair.setdefault(key, {})[r["condition"]] = r

    def score(record: dict, scale: str) -> float:
        value = scores.get(record["id"], {}).get(f"{scale}_score")
        return float("nan") if value is None else value  # NaN fails every comparison

    kept: set[str] = set()
    counts = {"pairs": 0, "kept_pairs": 0, "incomplete_pairs": 0}
    reasons = {"eliciting_trait_too_low": 0, "suppressing_trait_too_high": 0, "incoherent": 0}
    per_question: dict[int, int] = {}
    for key, pair in by_pair.items():
        if set(pair) != {"pos", "neg"}:
            counts["incomplete_pairs"] += 1
            continue
        counts["pairs"] += 1
        pos, neg = pair["pos"], pair["neg"]
        trait_ok = score(pos, "trait") >= trait_threshold
        suppressed_ok = score(neg, "trait") < 100 - trait_threshold
        coherent = (
            score(pos, "coherence") >= coherence_threshold
            and score(neg, "coherence") >= coherence_threshold
        )
        reasons["eliciting_trait_too_low"] += not trait_ok
        reasons["suppressing_trait_too_high"] += not suppressed_ok
        reasons["incoherent"] += not coherent
        if trait_ok and suppressed_ok and coherent:
            kept.update({pos["id"], neg["id"]})
            counts["kept_pairs"] += 1
            per_question[key[0]] = per_question.get(key[0], 0) + 1
    counts["kept_per_condition"] = counts["kept_pairs"]
    counts["dropped_for"] = reasons  # a pair can fail for more than one reason
    counts["kept_pairs_per_question"] = dict(sorted(per_question.items()))
    return kept, counts


def within_condition_correlation(
    projections: np.ndarray,
    trait_scores: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> list[dict]:
    """Inside one prompt condition, does a higher projection go with a higher
    judged trait score?

    All responses here were generated under trait-eliciting prompts, so the
    prompt cannot explain differences between them. ``projections`` is
    ``[responses, layers]``. Returns the Spearman correlation per layer with a
    bootstrap interval over questions.
    """
    rows = []
    for layer in range(projections.shape[1]):
        rho, low, high = spearman_with_ci(
            projections[:, layer], trait_scores, n_bootstrap, seed=seed, groups=groups
        )
        rows.append({"layer": layer, "spearman": rho, "spearman_low": low, "spearman_high": high})
    return rows
