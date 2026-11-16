"""Attribute the persona vector to the components that write it.

Decomposes the persona vector at the target layer into the direct contribution
of the token embeddings, of every attention layer and of every MLP neuron up to
that layer (see ``persona_neurons/attribution.py`` for the maths), then:

* asserts that all contributions add up to the length of the vector
  (the completeness check);
* reports how concentrated the neuron contributions are (cumulative share of
  the top ``k`` neurons, participation ratio), next to the same measures for
  random directions.

"Direct" means what a component adds to the residual stream itself. Effects
that pass through later components are not attributed, and no neuron is
ablated here, so a large contribution is not yet a causal claim.

It uses the extraction responses kept by ``scripts/extract_vector.py`` and
recomputes the vector from them in float32, because the decomposition is only
exact when no rounding happens between components. The recomputed vector must
match the saved one.

    uv run python scripts/attribute.py --config configs/sycophantic_olmo2_1b.yaml

Writes ``attribution/attribution.json`` and ``attribution/neurons.npz``.
float32 needs about 6 GB of GPU memory, so close other GPU programs first.
Smoke: seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import torch

from persona_neurons.attribution import (
    Attribution,
    check_completeness,
    component_means,
    concentration_summary,
    decompose,
    random_direction_concentration,
    top_neurons,
)
from persona_neurons.persona import build_prompt_ids
from persona_neurons.pipeline import (
    Context,
    load_kept_extraction,
    load_run_model,
    open_context,
    require,
)
from persona_neurons.stats import cosine
from persona_neurons.utils import load_json, run_metadata, save_json

# The vector is recomputed here in float32. It must point the same way as the
# one saved by extract_vector.py, which was computed in the generation dtype.
MIN_COSINE_WITH_SAVED_VECTOR = 0.99
# Number of points on the concentration curve, spaced evenly on a log scale.
N_CURVE_POINTS = 60


def means_by_condition(ctx: Context, model, tokenizer, records: list[dict]) -> dict:
    """Mean output of every component, for each condition separately."""
    att = ctx.cfg["attribution"]
    means = {}
    for condition in ("pos", "neg"):
        chosen = [r for r in records if r["condition"] == condition]
        prompts = [
            build_prompt_ids(
                tokenizer, r["system_prompt"], r["question"], ctx.cfg["prompts"]["prepend_bos"]
            )
            for r in chosen
        ]
        print(f"Collecting component means over {len(chosen)} {condition} responses ...")
        means[condition] = component_means(
            model,
            prompts,
            [r["response_ids"] for r in chosen],
            att["layer"],
            tokenizer.pad_token_id,
            att["batch_size"],
        )
    return means


def check_same_vector(ctx: Context, attribution: Attribution, model_info: dict) -> float:
    """Stop unless the decomposed vector is the one that was saved and tested.

    The completeness check cannot see this: it only ties the vector to the
    components recorded with it. Returns the cosine with the saved vector.
    """
    saved_commit = load_json(ctx.out / "model_info.json")["commit"]
    if saved_commit != model_info["commit"]:
        raise SystemExit(
            f"The saved vector comes from model commit {saved_commit}, but this run loaded "
            f"{model_info['commit']}. Pin model.revision in the config to a commit."
        )
    saved = torch.load(ctx.vectors_path)["vectors"][attribution.target_layer].numpy()
    similarity = cosine(attribution.vector, saved)
    if similarity < MIN_COSINE_WITH_SAVED_VECTOR:
        raise SystemExit(
            f"The recomputed vector has cosine {similarity:.4f} with the saved one. "
            "The responses or vectors on disk are out of date: rerun extract_vector.py."
        )
    return similarity


def concentration(model, attribution: Attribution, att: dict) -> dict:
    """Concentration of the neuron contributions, next to random directions."""
    n_neurons = attribution.neurons.size
    ks = tuple(
        sorted({int(k) for k in np.round(np.logspace(0, np.log10(n_neurons), N_CURVE_POINTS))})
    )
    persona = concentration_summary(attribution.neurons, ks)
    baseline = random_direction_concentration(
        model, attribution, att["n_random_directions"], att["seed"], ks
    )
    shares = np.array([[b["share_of_top_k"][k] for k in ks] for b in baseline])
    ratios = np.array([b["participation_ratio"] for b in baseline])

    def per_k(values: np.ndarray) -> dict:
        return dict(zip(ks, values.tolist(), strict=True))

    return {
        "k": list(ks),
        "persona": persona,
        "random_directions": {
            "n": len(baseline),
            "participation_ratio_mean": float(ratios.mean()),
            "participation_ratio_p2_5": float(np.quantile(ratios, 0.025)),
            "participation_ratio_p97_5": float(np.quantile(ratios, 0.975)),
            "share_of_top_k_mean": per_k(shares.mean(axis=0)),
            "share_of_top_k_p2_5": per_k(np.quantile(shares, 0.025, axis=0)),
            "share_of_top_k_p97_5": per_k(np.quantile(shares, 0.975, axis=0)),
        },
    }


def main() -> None:
    ctx, _ = open_context(__doc__.split("\n")[0])
    cfg, att = ctx.cfg, ctx.cfg["attribution"]
    require(ctx.out / "extract_kept.json", "extract_vector.py")
    out_dir = ctx.out / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loaded here in float32, not through ctx.model(), which uses the generation dtype.
    model, tokenizer, info = load_run_model(cfg, att["dtype"])
    records, _, _ = load_kept_extraction(ctx)
    means = means_by_condition(ctx, model, tokenizer, records)

    attribution = decompose(model, means["pos"], means["neg"], att["layer"])
    # Raises AssertionError with an explanation if the contributions do not add up.
    completeness = check_completeness(attribution, att["tolerance"])
    similarity = check_same_vector(ctx, attribution, info.to_dict())
    print(
        f"Completeness check passed: ||v|| = {attribution.vector_norm:.5f}, sum of contributions "
        f"= {attribution.total:.5f} (relative error {completeness['total_relative_error']:.1e})."
    )

    norm = attribution.vector_norm
    result = {
        "target_layer": att["layer"],
        "dtype": info.dtype,
        "n_eliciting": sum(r["condition"] == "pos" for r in records),
        "n_suppressing": sum(r["condition"] == "neg" for r in records),
        "vector_norm": norm,
        "cosine_with_saved_vector": similarity,
        "completeness": completeness,
        "share_of_vector": {
            "embeddings": attribution.embed / norm,
            "attention_total": float(attribution.attn.sum() / norm),
            "mlp_total": float(attribution.neurons.sum() / norm),
            "attention_by_layer": (attribution.attn / norm).tolist(),
            "mlp_by_layer": (attribution.neurons.sum(axis=1) / norm).tolist(),
        },
        "norm_scale_by_layer": {
            "eliciting": attribution.scale_pos.tolist(),
            "suppressing": attribution.scale_neg.tolist(),
        },
        "concentration": concentration(model, attribution, att),
        "top_neurons": top_neurons(attribution, att["n_top_neurons"]),
    }
    save_json(out_dir / "attribution.json", result, run_metadata(cfg, info.to_dict(), cfg["label"]))
    np.savez_compressed(
        out_dir / "neurons.npz",
        contribution=attribution.neurons,
        gap_with_scale=attribution.gap_with_scale,
        gap_without_scale=attribution.gap_without_scale,
        write_alignment=attribution.write_alignment,
        vector=attribution.vector,
    )

    shares, spread = result["share_of_vector"], result["concentration"]
    print(
        f"Share of the vector: embeddings {shares['embeddings']:+.3f}, attention "
        f"{shares['attention_total']:+.3f}, MLP neurons {shares['mlp_total']:+.3f}."
    )
    print(
        f"Participation ratio: {spread['persona']['participation_ratio']:.0f} of "
        f"{attribution.neurons.size} neurons (random directions: "
        f"{spread['random_directions']['participation_ratio_mean']:.0f})."
    )
    print(f"Outputs are in {out_dir}/")


if __name__ == "__main__":
    main()
