"""Score generated responses with the local judge.

Whether a response shows a trait is a judgement about meaning, so it is made
by a second, larger open-weight model. For every response the judge gives a
trait score and a coherence score from 0 to 100, using the official rubrics
of Chen et al. This script never loads OLMo, so the judge has the whole GPU.

``--input`` names the responses to score: ``extract`` and ``eval`` (from
``scripts/generate.py``) or ``steering`` (from ``scripts/steer.py``).

If the config lists several judges, each is loaded in turn (never two at
once), and the saved score is the average over the judges whose reply could be
parsed. Each judge's own score and raw reply are saved too.

    uv run python scripts/judge.py --config configs/sycophantic_olmo2_1b.yaml --input extract eval
    uv run python scripts/judge.py --config configs/sycophantic_olmo2_1b.yaml --input steering

Reads ``<input>_generations.jsonl`` and writes ``<input>_scores.jsonl``.
On an RTX 4070 Laptop GPU, a 7-8B judge in 4-bit used 5 to 7 GB of memory and
about half a second per response. Smoke: uses a fake judge, seconds on CPU.
"""

from __future__ import annotations

import time

import torch

from persona_neurons.judge import (
    failure_counts,
    load_coherence_rubric,
    make_judge,
    panel_scores,
    score_records,
)
from persona_neurons.pipeline import fingerprint, open_context, require, usable
from persona_neurons.utils import load_jsonl, save_jsonl

PRODUCED_BY = {"extract": "generate.py", "eval": "generate.py", "steering": "steer.py"}


def main() -> None:
    ctx, args = open_context(
        __doc__.split("\n")[0],
        lambda p: p.add_argument("--input", nargs="+", required=True, choices=list(PRODUCED_BY)),
    )
    cfg = ctx.cfg

    # The responses still to be judged, per input.
    pending = {}
    for name in args.input:
        gen_path = require(ctx.out / f"{name}_generations.jsonl", PRODUCED_BY[name])
        if ctx.needs(ctx.out / f"{name}_scores.jsonl"):
            pending[name] = usable(load_jsonl(gen_path))
        else:
            print(f"{name}: already judged. Use --force to judge again.")
    if not pending:
        return

    coherence_rubric = load_coherence_rubric(cfg["data"]["coherence_rubric_file"])
    per_judge = {name: {} for name in pending}
    judge_info = {name: {} for name in pending}
    for settings in cfg["judge"]["models"]:
        model_id = settings["model_id"]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        judge = make_judge(settings, cfg["smoke_mode"])  # one judge on the GPU at a time
        for name, records in pending.items():
            print(f"Judging {len(records)} {name} responses with {model_id} ...", flush=True)
            start = time.time()
            rows = score_records(judge, records, ctx.trait_rubric, coherence_rubric)
            per_judge[name][model_id] = rows
            judge_info[name][model_id] = {
                **judge.info,
                "seconds": round(time.time() - start, 1),
                "failures": failure_counts(rows),
                "peak_gpu_memory_gb": (
                    round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    if torch.cuda.is_available()
                    else None
                ),
            }
            print(f"    done in {judge_info[name][model_id]['seconds']} s")
        judge.close()  # free the GPU before the next judge

    for name in pending:
        rows = panel_scores(per_judge[name])
        scores_path = ctx.out / f"{name}_scores.jsonl"
        meta = {
            **ctx.meta(),
            "judges": judge_info[name],
            "generations_fingerprint": fingerprint(pending[name]),
        }
        save_jsonl(scores_path, rows, meta)
        unscored = sum(r["trait_score"] is None or r["coherence_score"] is None for r in rows)
        print(f"Wrote {scores_path}. Responses with a missing score: {unscored} of {len(rows)}.")


if __name__ == "__main__":
    main()
