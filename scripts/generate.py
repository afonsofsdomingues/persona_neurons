"""Generate the responses the persona vector is extracted from and tested on.

The model answers every question under two kinds of system prompt: one that
asks for the trait (trait-eliciting) and one that asks for its opposite
(trait-suppressing). Two sets of questions are used:

* the extraction questions, from which the vector is computed;
* the evaluation questions, held out to test the vector afterwards.

Several responses are sampled per prompt. Each is saved with its token ids, so
that later steps analyse exactly the tokens the model produced.

    uv run python scripts/generate.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``extract_generations.jsonl`` and ``eval_generations.jsonl`` (1600
responses with the shipped configs). Needs about 4 GB of GPU memory. Smoke:
seconds on CPU.
"""

from __future__ import annotations

from persona_neurons.pipeline import EVAL_SEED_OFFSET, Context, contrast_jobs, open_context
from persona_neurons.utils import save_jsonl


def generate_set(ctx: Context, name: str, questions: list[str], settings: dict, seed: int) -> None:
    """Generate both conditions for one set of questions and save them."""
    path = ctx.out / f"{name}_generations.jsonl"
    if not ctx.needs(path):
        print(f"{path} exists. Use --force to generate again.")
        return
    n_questions = ctx.cfg.get("smoke_n_questions", len(questions))
    jobs = contrast_jobs(
        ctx,
        name,
        list(range(n_questions)),
        settings["n_instruction_pairs"],
        settings["samples_per_prompt"],
        questions=questions,
    )
    ctx.generate(jobs, seed=seed)
    save_jsonl(path, jobs, ctx.meta())
    print(f"Generated {len(jobs)} responses -> {path}")


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    cfg = ctx.cfg
    seed = cfg["generation"]["seed"]
    generate_set(ctx, "extract", ctx.questions, cfg["extraction"], seed)
    generate_set(ctx, "eval", ctx.eval_questions, cfg["separation"], seed + EVAL_SEED_OFFSET)


if __name__ == "__main__":
    main()
