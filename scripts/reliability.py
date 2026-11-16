"""Split-half reliability of the persona vector.

The extraction questions are divided into two disjoint halves, a vector is
extracted from each half, and the cosine similarity between the two is
reported at every layer. If the vector reflects the trait and not the
particular questions, the two halves should agree.

Uses the responses kept by ``scripts/extract_vector.py``. No model is loaded.

    uv run python scripts/reliability.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``reliability.json``. Runtime: seconds, CPU only.
"""

from __future__ import annotations

from persona_neurons.analysis import reliability_table
from persona_neurons.pipeline import arrays, load_kept_extraction, open_context, require
from persona_neurons.utils import save_json


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    require(ctx.out / "extract_kept.json", "extract_vector.py")
    records, acts, _ = load_kept_extraction(ctx)
    is_pos, groups = arrays(records)

    cfg = ctx.cfg["reliability"]
    halves, rows = reliability_table(acts, is_pos, groups, cfg["seed"], cfg["n_random_splits"])
    save_json(ctx.out / "reliability.json", {**halves, "layers": rows}, ctx.meta())

    print("Layer  split-half cosine  [range over random splits]")
    for row in rows:
        print(
            f"{row['layer']:5d}  {row['cosine']:.3f}  "
            f"[{row['cosine_random_min']:.3f}, {row['cosine_random_max']:.3f}]"
        )


if __name__ == "__main__":
    main()
