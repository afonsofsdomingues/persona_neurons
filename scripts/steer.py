"""Steer the model with the persona vector on the evaluation questions.

The writing test. Under the neutral system prompt, the model answers each
evaluation question while

    alpha * mean_residual_norm(layer) * v_hat(layer)

is added to the residual stream after one layer. ``alpha`` is therefore a
fraction of the residual stream's own typical length. The layers, the grid of
``alpha`` values (negative values and zero included) and the positions at
which the vector is added come from the ``steering`` block of the config.

Every generation is saved with its settings. Nothing is scored here:
``scripts/judge.py --input steering`` and ``scripts/steering_summary.py``
come next.

    uv run python scripts/steer.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``steering_generations.jsonl`` (3100 responses with the shipped
configs, the longest step). Needs about 4 GB of GPU memory. Smoke: seconds on
CPU.
"""

from __future__ import annotations

from persona_neurons.pipeline import open_context, require, run_steering, vectors_fingerprint
from persona_neurons.utils import load_json


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    require(ctx.vectors_path, "extract_vector.py")
    path = ctx.out / "steering_generations.jsonl"
    if not ctx.needs(path):
        steered_with = load_json(path.with_suffix(".meta.json")).get("vectors_fingerprint")
        if steered_with != vectors_fingerprint(ctx.vectors_path):
            raise SystemExit(f"{path} was steered with other vectors. Rerun with --force.")
        print(f"{path} exists. Use --force to generate again.")
        return
    n_questions = ctx.cfg.get("smoke_n_questions", len(ctx.eval_questions))
    questions = list(enumerate(ctx.eval_questions))[:n_questions]
    run_steering(ctx, ctx.cfg["steering"]["layers"], questions, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
