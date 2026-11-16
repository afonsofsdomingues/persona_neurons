"""Exact attribution of a persona vector to the components that write it.

The claim
---------
A persona vector at layer ``L`` is a difference of mean residual-stream
activations between two sets of responses:

    v = mean_pos[x_L] - mean_neg[x_L]

In OLMo 2 the residual stream is a plain sum of component outputs:

    x_L = embed + sum_{l<=L} attn_l + sum_{l<=L} mlp_l

where ``attn_l`` and ``mlp_l`` are the *normalised* sublayer outputs (the
normalisation sits on the sublayer output, not on the residual). Means and
differences are linear,
so ``v`` splits into one difference per component. Projecting each onto the
unit vector ``v_hat = v / ||v||`` gives numbers that add up to the length of
the vector:

    ||v|| = v . v_hat = sum over components c of (mean_pos[c] - mean_neg[c]) . v_hat

Each term is that component's *contribution*: how much of the persona vector
it writes. A negative contribution means the component pushes against it.

Down to single neurons
----------------------
A normalised MLP output is a sum over the layer's neurons. With ``a_i`` the
activation of neuron ``i``, ``w_i`` its output weights, ``s`` the per-token
scale of the layer's output norm and ``g`` that norm's learned gain:

    mlp(t) = sum_i  s(t) * a_i(t) * (g * w_i)

so the contribution of neuron ``i`` is a product of two factors:

    C_i = ( mean_pos[s * a_i] - mean_neg[s * a_i] )  *  ( (g * w_i) . v_hat )
          \\__________ activation gap ___________/     \\__ write alignment _/

* The **activation gap** says how differently the neuron fires in the two
  conditions. A neuron that is always on has no gap and contributes nothing,
  however well its weights align with the vector.
* The **write alignment** says how much one unit of the neuron's activation
  moves the residual stream along the persona direction. It depends only on
  weights.

The shared scale ``s``
----------------------
``s(t)`` is computed from the whole MLP output, so it depends on every neuron
in the layer. A neuron's contribution can therefore change when only its
neighbours change. The gap is reported twice: *with* ``s`` (the exact
contribution) and *without* it (``mean_pos[a_i] - mean_neg[a_i]``, the neuron
alone). Only the version with ``s`` adds up exactly.

Completeness
------------
All contributions must sum to ``||v||``. :func:`check_completeness` asserts
this. If the normalisation were handled wrongly anywhere, the sum would be
off and the check would fail.

The check ties the vector to the components recorded in the same forward
passes. It cannot tell whether those passes used the intended responses, so
``scripts/attribute.py`` separately compares the vector with the saved one.

Numerical precision: run this with the model in float32. In bfloat16 the
model rounds after every addition, and the sum is only approximate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from persona_neurons.baselines import random_unit_directions
from persona_neurons.hooks import ComponentCapture, decoder_layers
from persona_neurons.persona import batches, teacher_forcing_batch

# --------------------------------------------------------------------------- #
# Step 1: mean component outputs per condition
# --------------------------------------------------------------------------- #


@torch.no_grad()
def component_means(
    model,
    prompts: list[list[int]],
    responses: list[list[int]],
    last_layer: int,
    pad_id: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Mean output of every component over a set of responses.

    Each response is first averaged over its own response tokens, then the
    responses are averaged with equal weight. This is the same averaging the
    persona vector uses, which is what makes the decomposition exact.

    Returns float64 arrays: ``embed`` ``[hidden]``, ``attn`` and ``mlp``
    ``[layers, hidden]``, ``neurons_raw`` and ``neurons_scaled``
    ``[layers, neurons]``, ``scale`` ``[layers]`` and ``residual`` ``[hidden]``,
    where ``layers = last_layer + 1``.
    """
    n_layers = last_layer + 1
    totals: dict[str, torch.Tensor] = {}

    def accumulate(name: str, batch_rows: torch.Tensor) -> None:
        summed = batch_rows.double().sum(dim=0).cpu()
        totals[name] = totals[name] + summed if name in totals else summed

    with ComponentCapture(model, last_layer) as capture:
        for index_range in batches(len(prompts), batch_size):
            input_ids, attention_mask, response_mask = teacher_forcing_batch(
                [prompts[i] for i in index_range], [responses[i] for i in index_range], pad_id
            )
            capture.set_mask(response_mask.to(model.device))
            model(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device)
            )
            accumulate("embed", capture.embed)
            accumulate("residual", capture.residual)
            for name in ("attn", "mlp", "neurons_raw", "neurons_scaled", "scale"):
                rows = torch.stack([getattr(capture, name)[layer] for layer in range(n_layers)], 1)
                accumulate(name, rows)  # rows: [batch, layers, ...]
    return {name: (total / len(prompts)).numpy() for name, total in totals.items()}


# --------------------------------------------------------------------------- #
# Step 2: the decomposition
# --------------------------------------------------------------------------- #


@dataclass
class Attribution:
    """Result of decomposing one persona vector. ``layers = target_layer + 1``."""

    target_layer: int
    vector: np.ndarray  # [hidden]  v at the target layer
    vector_norm: float  # ||v||
    embed: float  # contribution of the token embeddings
    attn: np.ndarray  # [layers]  contribution of each attention layer
    mlp: np.ndarray  # [layers]  contribution of each MLP layer as a whole
    neurons: np.ndarray  # [layers, neurons]  contribution of each neuron
    gap_with_scale: np.ndarray  # [layers, neurons]  mean_pos[s a] - mean_neg[s a]
    gap_without_scale: np.ndarray  # [layers, neurons]  mean_pos[a] - mean_neg[a]
    write_alignment: np.ndarray  # [layers, neurons]  (g * w_i) . v_hat
    scale_pos: np.ndarray  # [layers]  mean of s over trait-eliciting responses
    scale_neg: np.ndarray  # [layers]  mean of s over trait-suppressing responses

    @property
    def total(self) -> float:
        """Sum of all contributions. Should equal ``vector_norm``."""
        return float(self.embed + self.attn.sum() + self.neurons.sum())


def output_weights(model, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """``(W, g)`` for one layer as float64: ``W`` is ``down_proj.weight``
    ``[hidden, neurons]`` (column ``i`` is neuron ``i``'s output weights) and
    ``g`` is the gain of the norm applied to the MLP output ``[hidden]``."""
    block = decoder_layers(model)[layer]
    weight = block.mlp.down_proj.weight.detach().double().cpu().numpy()
    gain = block.post_feedforward_layernorm.weight.detach().double().cpu().numpy()
    return weight, gain


def write_alignments(model, last_layer: int, direction: np.ndarray) -> np.ndarray:
    """Write alignment of every neuron in layers ``0..last_layer`` with a unit
    direction: ``(g * w_i) . direction``. Returns ``[layers, neurons]``.

    For all neurons of a layer at once this is ``W^T (g * direction)``.
    """
    rows = []
    for layer in range(last_layer + 1):
        weight, gain = output_weights(model, layer)
        rows.append(weight.T @ (gain * direction))
    return np.stack(rows)


def decompose(model, means_pos: dict, means_neg: dict, target_layer: int) -> Attribution:
    """Split the persona vector at ``target_layer`` into contributions.

    ``means_pos`` and ``means_neg`` come from :func:`component_means` for the
    trait-eliciting and trait-suppressing responses.
    """
    vector = means_pos["residual"] - means_neg["residual"]
    norm = float(np.linalg.norm(vector))
    direction = vector / norm  # v_hat

    alignment = write_alignments(model, target_layer, direction)
    gap_with_scale = means_pos["neurons_scaled"] - means_neg["neurons_scaled"]
    return Attribution(
        target_layer=target_layer,
        vector=vector,
        vector_norm=norm,
        embed=float((means_pos["embed"] - means_neg["embed"]) @ direction),
        attn=(means_pos["attn"] - means_neg["attn"]) @ direction,
        mlp=(means_pos["mlp"] - means_neg["mlp"]) @ direction,
        neurons=gap_with_scale * alignment,
        gap_with_scale=gap_with_scale,
        gap_without_scale=means_pos["neurons_raw"] - means_neg["neurons_raw"],
        write_alignment=alignment,
        scale_pos=means_pos["scale"],
        scale_neg=means_neg["scale"],
    )


def check_completeness(attribution: Attribution, tolerance: float = 1e-3) -> dict:
    """Assert that the contributions add up to the length of the vector.

    Two checks, both relative to ``||v||``:

    1. embeddings + attention layers + all neurons = ``||v||``;
    2. within every layer, the neurons sum to the contribution of the whole
       (normalised) MLP output measured directly.

    Returns the two errors. Raises ``AssertionError`` with an explanation if
    either exceeds ``tolerance``.
    """
    norm = attribution.vector_norm
    total_error = abs(attribution.total - norm) / norm
    layer_errors = np.abs(attribution.neurons.sum(axis=1) - attribution.mlp) / norm
    worst_layer = int(layer_errors.argmax())
    if total_error > tolerance or layer_errors.max() > tolerance:
        raise AssertionError(
            "Attribution is not complete: the contributions do not add up to the persona "
            f"vector.\n  ||v|| = {norm:.6g}, sum of contributions = {attribution.total:.6g} "
            f"(relative error {total_error:.2e}, tolerance {tolerance:.0e}).\n  Worst layer: "
            f"neurons of layer {worst_layer} sum to {attribution.neurons[worst_layer].sum():.6g} "
            f"but its MLP output contributes {attribution.mlp[worst_layer]:.6g} (relative error "
            f"{layer_errors.max():.2e}).\n  Likely causes: the model is not in float32, the "
            "normalisation formula or its placement differs from what this "
            "module assumes."
        )
    return {
        "total_relative_error": float(total_error),
        "worst_layer_relative_error": float(layer_errors.max()),
        "tolerance": tolerance,
    }


# --------------------------------------------------------------------------- #
# Step 3: concentration
# --------------------------------------------------------------------------- #


def participation_ratio(contributions: np.ndarray) -> float:
    """Effective number of neurons that carry the contributions.

        PR = (sum_i |C_i|)^2 / sum_i C_i^2

    If ``k`` neurons contribute equally and the rest nothing, PR = ``k``. One
    dominant neuron gives a value near 1; all neurons contributing equally
    gives the number of neurons.
    """
    magnitudes = np.abs(np.asarray(contributions, dtype=float).ravel())
    return float(magnitudes.sum() ** 2 / (np.square(magnitudes).sum() + 1e-300))


def cumulative_share(contributions: np.ndarray) -> np.ndarray:
    """Share of the total absolute contribution held by the top ``k`` neurons,
    for every ``k``.

    Neurons are sorted by ``|C_i|``, largest first. Entry ``k-1`` of the result
    is ``sum of the k largest |C_i|`` divided by ``sum of all |C_i|``. The curve
    rises from near 0 to 1. A steep early rise means few neurons do most of
    the writing.
    """
    magnitudes = np.sort(np.abs(np.asarray(contributions, dtype=float).ravel()))[::-1]
    return np.cumsum(magnitudes) / (magnitudes.sum() + 1e-300)


def concentration_summary(contributions: np.ndarray, top_k: tuple[int, ...]) -> dict:
    """Participation ratio and the cumulative share at a few values of ``k``."""
    curve = cumulative_share(contributions)
    return {
        "n_neurons": int(curve.size),
        "participation_ratio": participation_ratio(contributions),
        "share_of_top_k": {int(k): float(curve[min(k, curve.size) - 1]) for k in top_k},
    }


def random_direction_concentration(
    model,
    attribution: Attribution,
    n_directions: int,
    seed: int,
    top_k: tuple[int, ...],
) -> list[dict]:
    """The same concentration measures for random directions.

    A concentration number means little alone: neuron activations are
    heavy-tailed, so *any* direction will look somewhat concentrated. For each
    random unit direction ``r`` we keep the activation gaps exactly as they are
    and replace only the write alignment:

        C_i(r) = gap_i * ( (g * w_i) . r )

    This is the exact decomposition of ``v . r``, the component of the same
    persona contrast along ``r``. The direction has unit length, like
    ``v_hat``; scaling it would scale all contributions equally and leave the
    concentration unchanged.

    What this baseline can show. The persona vector is itself built from the
    gaps: ``v`` contains the term ``gap_i * (g * w_i)`` for every neuron, so a
    neuron with a large gap automatically has some positive alignment with
    ``v``, which no random direction has. Some extra concentration along ``v``
    is therefore expected for *any* difference-of-means vector. Being more
    concentrated than random directions is a necessary check, not evidence
    that persona vectors are special. The matched comparison, pushing other
    difference-of-means vectors (shuffled labels, unrelated contrasts) through
    the same decomposition, is not implemented yet.
    """
    last_layer = attribution.target_layer
    weights = [output_weights(model, layer) for layer in range(last_layer + 1)]
    summaries = []
    for direction in random_unit_directions(n_directions, attribution.vector.size, seed):
        alignment = np.stack([weight.T @ (gain * direction) for weight, gain in weights])
        summaries.append(concentration_summary(attribution.gap_with_scale * alignment, top_k))
    return summaries


def top_neurons(attribution: Attribution, n: int) -> list[dict]:
    """The ``n`` neurons with the largest absolute contribution, with both
    versions of the activation gap and the write alignment."""
    flat = np.abs(attribution.neurons).ravel()
    order = np.argsort(flat)[::-1][:n]
    n_neurons = attribution.neurons.shape[1]
    rows = []
    for rank, flat_index in enumerate(order, start=1):
        layer, neuron = divmod(int(flat_index), n_neurons)
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "neuron": neuron,
                "contribution": float(attribution.neurons[layer, neuron]),
                "share_of_vector": float(
                    attribution.neurons[layer, neuron] / attribution.vector_norm
                ),
                "gap_with_scale": float(attribution.gap_with_scale[layer, neuron]),
                "gap_without_scale": float(attribution.gap_without_scale[layer, neuron]),
                "write_alignment": float(attribution.write_alignment[layer, neuron]),
            }
        )
    return rows
