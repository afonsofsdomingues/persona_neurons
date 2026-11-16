"""Does the persona vector separate held-out responses?

The reading test. Every response to the evaluation questions gets one number
per layer: the mean projection of its response tokens on the persona vector.
If the vector reads the trait, trait-eliciting responses should score higher
than trait-suppressing ones. How well the two sets separate is measured by the
AUROC (1.0 = perfectly, 0.5 = not at all), with a bootstrap interval over
questions.

Three results per layer:

* **Prompt-controlled (primary).** Responses are re-encoded under one neutral
  system prompt, so only what the response says can separate the two sets.
* **Same prompts (for comparison).** Responses are encoded under the prompts
  that generated them. The prompt alone can produce this separation.
* **Baselines.** Shuffled-label vectors and random directions through the same
  prompt-controlled computation.

If the evaluation responses have been judged, it also reports, inside the
trait-eliciting condition only, the Spearman correlation between projection
and judged trait score.

    uv run python scripts/separation.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``separation.json``. Needs about 4 GB of GPU memory. Smoke: seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from persona_neurons.analysis import (
    numbered_list_fraction,
    separation_table,
    within_condition_correlation,
)
from persona_neurons.persona import project
from persona_neurons.pipeline import (
    arrays,
    load_acts,
    load_kept_extraction,
    load_scores,
    open_context,
    require,
    usable,
)
from persona_neurons.utils import load_jsonl, save_json


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    cfg = ctx.cfg
    sep_cfg = cfg["separation"]
    require(ctx.vectors_path, "extract_vector.py")

    gen_path = require(ctx.out / "eval_generations.jsonl", "generate.py")
    generated = load_jsonl(gen_path)
    records = usable(generated)

    own_path, neutral_path = ctx.cache / "eval_acts_own.pt", ctx.cache / "eval_acts_neutral.pt"
    if ctx.needs(own_path):
        torch.save(ctx.activations(records, system="own"), own_path)
    if ctx.needs(neutral_path):
        torch.save(ctx.activations(records, system=ctx.neutral), neutral_path)
    acts_own, _ = load_acts(own_path, records, ctx.acts_fingerprint(records, "own"))
    acts_neutral, _ = load_acts(neutral_path, records, ctx.acts_fingerprint(records, ctx.neutral))
    is_pos, groups = arrays(records)

    # ---- Separation table --------------------------------------------------------
    saved = torch.load(ctx.vectors_path)
    vectors = {k: v.numpy() for k, v in saved.items() if k != "meta"}
    extract_records, extract_acts, _ = load_kept_extraction(ctx)
    extract_pos, extract_groups = arrays(extract_records)
    rows = separation_table(
        vectors,
        extract_acts,
        extract_pos,
        extract_groups,
        acts_controlled=acts_neutral,
        acts_own=acts_own,
        is_pos_test=is_pos,
        groups_test=groups,
        n_bootstrap=sep_cfg["n_bootstrap"],
        n_baseline=sep_cfg["n_shuffles"],
        seed=cfg["seeds"]["baselines"],
    )
    result = {
        "counts": {
            "eval_generated": len(generated),
            "eval_used_eliciting": int(is_pos.sum()),
            "eval_used_suppressing": int((~is_pos).sum()),
            "numbered_list_fraction_eliciting": numbered_list_fraction(records, "pos"),
            "numbered_list_fraction_suppressing": numbered_list_fraction(records, "neg"),
        },
        "layers": rows,
    }

    # ---- Within the trait-eliciting condition, if judged ---------------------------
    scores_path = ctx.out / "eval_scores.jsonl"
    judged = scores_path.exists()
    if judged:
        scores = load_scores(scores_path, records)
        trait = np.array(
            [
                np.nan if scores[r["id"]]["trait_score"] is None else scores[r["id"]]["trait_score"]
                for r in records
            ]
        )
        use = is_pos & ~np.isnan(trait)
        directions = vectors["unit_vectors"]
    if judged and use.sum() >= 3:  # a rank correlation needs a few scored responses
        seed = cfg["seeds"]["baselines"]
        result["within_eliciting_condition"] = {
            "n_responses": int(use.sum()),
            "own_prompts": within_condition_correlation(
                project(acts_own[use], directions),
                trait[use],
                groups[use],
                sep_cfg["n_bootstrap"],
                seed,
            ),
            "neutral_prompt": within_condition_correlation(
                project(acts_neutral[use], directions),
                trait[use],
                groups[use],
                sep_cfg["n_bootstrap"],
                seed,
            ),
        }
    save_json(ctx.out / "separation.json", result, ctx.meta())

    print("Layer  prompt-controlled AUROC [95% CI]   same prompts   random directions 97.5th pct")
    for r in rows:
        print(
            f"{r['layer']:5d}  {r['auroc_controlled']:.3f} [{r['auroc_controlled_low']:.3f}, "
            f"{r['auroc_controlled_high']:.3f}]   {r['auroc_own']:.3f}   {r['random_p97_5']:.3f}"
        )
    if "within_eliciting_condition" not in result:
        print("\nToo few judged trait-eliciting responses: the within-condition")
        print("correlation was skipped.")


if __name__ == "__main__":
    main()
