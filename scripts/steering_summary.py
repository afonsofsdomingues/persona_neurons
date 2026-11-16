"""Does steering with the persona vector produce the trait?

The result of the writing test. ``scripts/steer.py`` generated answers with
the vector added at several layers and strengths (``alpha``), and
``scripts/judge.py`` scored them. This script averages the scores for every
setting:

* the mean judged trait score and the mean coherence, each with a bootstrap
  interval over questions;
* the change in trait score from the unsteered answers (``alpha = 0``),
  compared question by question;
* two judge-free signs of broken text: the share of empty answers and the
  share of repeated 3-grams (text stuck in a loop).

A vector that writes the trait should raise the trait score as ``alpha`` grows
while coherence stays high. As a one-number summary per layer, the script
reports the Spearman correlation between ``alpha`` and the mean trait score.

    uv run python scripts/steering_summary.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``steering.json``. Seconds, CPU only.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from persona_neurons.persona import repeated_ngram_fraction
from persona_neurons.pipeline import Context, load_scores, open_context, require, usable
from persona_neurons.stats import mean_with_ci, spearman
from persona_neurons.utils import load_jsonl, save_json


def summarise_setting(
    ctx: Context, records: list[dict], scores: dict, unsteered_trait: dict[int, float]
) -> dict:
    """Average scores of the answers generated under one ``(layer, alpha)``.

    ``unsteered_trait`` maps a question to its mean trait score at
    ``alpha = 0``, so the increase is measured against the same question.
    """
    n_boot, seed = ctx.cfg["steering"]["n_bootstrap"], ctx.cfg["generation"]["seed"]
    repeats = [repeated_ngram_fraction(r["response_ids"]) for r in records]
    repeats = [value for value in repeats if not np.isnan(value)]
    scored = [r for r in usable(records) if scores[r["id"]]["trait_score"] is not None]
    coherent = [r for r in usable(records) if scores[r["id"]]["coherence_score"] is not None]
    row = {
        "layer": records[0]["layer"],
        "alpha": records[0]["alpha"],
        "n_responses": len(records),
        "empty_fraction": 1 - len(usable(records)) / len(records),
        "repeated_3gram_fraction": float(np.mean(repeats)) if repeats else float("nan"),
        "n_trait_scored": len(scored),
        "n_coherence_scored": len(coherent),
    }
    if scored:
        trait = np.array([scores[r["id"]]["trait_score"] for r in scored])
        questions = np.array([r["question_idx"] for r in scored])
        row["trait"], row["trait_low"], row["trait_high"] = mean_with_ci(
            trait, questions, n_boot, seed
        )
        if unsteered_trait:
            known = np.array([q in unsteered_trait for q in questions])
            change = trait[known] - np.array([unsteered_trait[q] for q in questions[known]])
            row["trait_increase"], row["trait_increase_low"], row["trait_increase_high"] = (
                mean_with_ci(change, questions[known], n_boot, seed)
            )
    if coherent:
        coherence = np.array([scores[r["id"]]["coherence_score"] for r in coherent])
        questions = np.array([r["question_idx"] for r in coherent])
        row["coherence"], row["coherence_low"], row["coherence_high"] = mean_with_ci(
            coherence, questions, n_boot, seed
        )
    return row


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    records = load_jsonl(require(ctx.out / "steering_generations.jsonl", "steer.py"))
    scores_path = require(ctx.out / "steering_scores.jsonl", "judge.py --input steering")
    scores = load_scores(scores_path, usable(records))

    by_setting = defaultdict(list)
    for record in records:
        by_setting[(record["layer"], record["alpha"])].append(record)

    # Mean trait score of each question without steering (layer None, alpha 0).
    unsteered = defaultdict(list)
    for record in usable(by_setting[(None, 0.0)]):
        score = scores[record["id"]]["trait_score"]
        if score is not None:
            unsteered[record["question_idx"]].append(score)
    unsteered_trait = {question: float(np.mean(values)) for question, values in unsteered.items()}

    baseline = summarise_setting(ctx, by_setting[(None, 0.0)], scores, {})
    layers = []
    for layer in ctx.cfg["steering"]["layers"]:
        rows = [
            summarise_setting(ctx, by_setting[key], scores, unsteered_trait)
            for key in sorted(k for k in by_setting if k[0] == layer)
        ]
        rows = sorted([*rows, baseline], key=lambda row: row["alpha"])
        with_trait = [row for row in rows if "trait" in row]
        layers.append(
            {
                "layer": layer,
                "spearman_alpha_trait": spearman(
                    [row["alpha"] for row in with_trait], [row["trait"] for row in with_trait]
                ),
                "settings": rows,
            }
        )
    save_json(ctx.out / "steering.json", {"layers": layers}, ctx.meta())

    def cell(row: dict, key: str) -> str:
        return f"{row[key]:5.1f}" if key in row else "    -"

    for entry in layers:
        print(
            f"\nLayer {entry['layer']}: Spearman(alpha, trait) = "
            f"{entry['spearman_alpha_trait']:.2f}"
        )
        print("  alpha   trait  increase  coherence  empty  repeated 3-grams")
        for row in entry["settings"]:
            print(
                f"  {row['alpha']:+.2f}   {cell(row, 'trait')}     {cell(row, 'trait_increase')}"
                f"      {cell(row, 'coherence')}   {row['empty_fraction']:.2f}  "
                f"{row['repeated_3gram_fraction']:.3f}"
            )


if __name__ == "__main__":
    main()
