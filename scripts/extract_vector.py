"""Compute the persona vector at every layer.

Takes the responses to the extraction questions (``scripts/generate.py``) and
their judge scores (``scripts/judge.py``), and:

1. records the model's residual stream on every response, averaged over the
   response tokens;
2. keeps only the response pairs in which the prompts worked, as Chen et al.
   do: the trait-eliciting response shows the trait, the trait-suppressing one
   does not, and both are coherent;
3. subtracts the two condition means of the kept responses at every layer ``l``:

       v_l = mean over eliciting responses [x_l] - mean over suppressing responses [x_l]

With ``filter.enabled: false`` in the config, step 2 is skipped and no judge
scores are needed.

    uv run python scripts/extract_vector.py --config configs/sycophantic_olmo2_1b.yaml

Outputs in the run's folder:

* ``extract_kept.json``     which responses were kept, and why others were not
* ``persona_vectors.pt``    vectors, unit vectors, their norms, mean residual norms
* ``vectors_summary.json``  the per-layer numbers in readable form

Needs about 4 GB of GPU memory. Smoke: seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from persona_neurons.analysis import filter_pairs, persona_vectors
from persona_neurons.pipeline import (
    arrays,
    load_acts,
    load_scores,
    open_context,
    require,
    usable,
)
from persona_neurons.stats import cosine
from persona_neurons.utils import load_jsonl, save_json


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    cfg = ctx.cfg

    gen_path = require(ctx.out / "extract_generations.jsonl", "generate.py")
    generated = load_jsonl(gen_path)
    records = usable(generated)
    filter_cfg = cfg["filter"]
    scores_path = ctx.out / "extract_scores.jsonl"
    if filter_cfg["enabled"]:
        require(scores_path, "judge.py --input extract")

    # ---- Activations for every usable response ---------------------------------
    acts_path = ctx.cache / "extract_acts.pt"
    if ctx.needs(acts_path):
        torch.save(ctx.activations(records, system="own"), acts_path)
    acts, norms = load_acts(acts_path, records, ctx.acts_fingerprint(records, "own"))
    is_pos, _ = arrays(records)

    # ---- Filter ------------------------------------------------------------------
    if filter_cfg["enabled"]:
        kept_ids, counts = filter_pairs(
            records,
            load_scores(scores_path, records),
            filter_cfg["trait_threshold"],
            filter_cfg["coherence_threshold"],
        )
    else:
        kept_ids, counts = {r["id"] for r in records}, {"note": "filtering disabled"}
    keep = np.array([r["id"] in kept_ids for r in records])
    counts["generated"] = len(generated)
    counts["usable_eliciting"] = int(is_pos.sum())
    counts["usable_suppressing"] = int((~is_pos).sum())
    counts["kept_eliciting"] = int((keep & is_pos).sum())
    counts["kept_suppressing"] = int((keep & ~is_pos).sum())
    if counts["kept_eliciting"] == 0 or counts["kept_suppressing"] == 0:
        raise SystemExit(f"No responses survived the filter. Counts: {counts}")
    save_json(
        ctx.out / "extract_kept.json",
        {"filter": filter_cfg, "counts": counts, "kept_ids": sorted(kept_ids)},
        ctx.meta(),
    )

    # ---- Vectors -----------------------------------------------------------------
    vectors = persona_vectors(acts[keep], norms[keep], is_pos[keep])
    unfiltered = persona_vectors(acts, norms, is_pos)
    torch.save(
        {"meta": ctx.meta(), **{k: torch.tensor(v) for k, v in vectors.items()}},
        ctx.vectors_path,
    )
    layers = [
        {
            "layer": layer,
            "vector_norm": float(vectors["vector_norm"][layer]),
            "mean_residual_norm": float(vectors["mean_residual_norm"][layer]),
            "vector_norm_over_residual_norm": float(
                vectors["vector_norm"][layer] / vectors["mean_residual_norm"][layer]
            ),
            "cosine_with_unfiltered_vector": cosine(
                vectors["vectors"][layer], unfiltered["vectors"][layer]
            ),
        }
        for layer in range(vectors["vectors"].shape[0])
    ]
    save_json(ctx.out / "vectors_summary.json", {"counts": counts, "layers": layers}, ctx.meta())

    print(f"\nResponses kept per condition: {counts['kept_eliciting']} eliciting, "
          f"{counts['kept_suppressing']} suppressing (of {counts['usable_eliciting']} and "
          f"{counts['usable_suppressing']} usable).")  # fmt: skip
    if filter_cfg["enabled"]:
        print(f"Pairs dropped for: {counts['dropped_for']}")
    print(f"Vectors saved to {ctx.vectors_path}")


if __name__ == "__main__":
    main()
