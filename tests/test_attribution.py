"""Tests for the exact attribution, on a tiny random OLMo 2 model (CPU, seconds)."""

import dataclasses

import numpy as np
import pytest
import torch

from persona_neurons.attribution import (
    check_completeness,
    component_means,
    cumulative_share,
    decompose,
    participation_ratio,
    random_direction_concentration,
    top_neurons,
)
from persona_neurons.hooks import ResidualCapture
from persona_neurons.loading import load_tiny_model
from persona_neurons.persona import mean_response_activations

# Two small sets of (prompt, response) token ids of unequal lengths.
POS = ([[5, 6, 7, 8], [9, 10], [11, 12, 13]], [[20, 21, 22], [30, 31, 32, 33, 34], [40, 41]])
NEG = ([[5, 6, 7, 8], [9, 10], [11, 12, 13]], [[50, 51], [60, 61, 62], [70, 71, 72, 73]])


@pytest.fixture(scope="module")
def setup():
    model, tokenizer, _ = load_tiny_model(seed=0)
    target_layer = 2
    means_pos = component_means(model, *POS, target_layer, tokenizer.pad_token_id, batch_size=2)
    means_neg = component_means(model, *NEG, target_layer, tokenizer.pad_token_id, batch_size=2)
    return model, tokenizer, target_layer, decompose(model, means_pos, means_neg, target_layer)


def test_contributions_sum_to_the_vector_norm(setup):
    _, _, _, attribution = setup
    errors = check_completeness(attribution, tolerance=1e-4)
    assert errors["total_relative_error"] < 1e-4
    assert errors["worst_layer_relative_error"] < 1e-4
    assert attribution.total == pytest.approx(attribution.vector_norm, rel=1e-4)


def test_the_decomposed_vector_is_the_persona_vector(setup):
    """The vector being decomposed must equal the usual difference of means."""
    model, tokenizer, target_layer, attribution = setup
    acts_pos, _ = mean_response_activations(model, *POS, tokenizer.pad_token_id, batch_size=3)
    acts_neg, _ = mean_response_activations(model, *NEG, tokenizer.pad_token_id, batch_size=3)
    expected = acts_pos[:, target_layer].mean(0) - acts_neg[:, target_layer].mean(0)
    assert np.allclose(attribution.vector, expected.numpy(), atol=1e-5)


def test_neuron_contribution_is_gap_times_alignment(setup):
    _, _, _, attribution = setup
    product = attribution.gap_with_scale * attribution.write_alignment
    assert np.allclose(attribution.neurons, product)
    # The gap without the shared scale is a different quantity.
    assert not np.allclose(attribution.gap_with_scale, attribution.gap_without_scale)


def test_scaled_neuron_outputs_rebuild_the_normalised_mlp_output(setup):
    """Token by token: sum_i s * a_i * (g * w_i) equals the layer's normalised MLP output."""
    model, _, _, _ = setup
    ids = torch.tensor([[3, 4, 5, 6, 7, 8]])
    layer = model.model.layers[1]
    seen = {}
    handles = [
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, args: seen.__setitem__("a", args[0])
        ),
        layer.mlp.register_forward_hook(lambda m, args, out: seen.__setitem__("m", out)),
        layer.post_feedforward_layernorm.register_forward_hook(
            lambda m, args, out: seen.__setitem__("normed", out)
        ),
    ]
    with torch.no_grad():
        model(ids)
    for handle in handles:
        handle.remove()
    s = torch.rsqrt(seen["m"].pow(2).mean(-1, keepdim=True) + model.config.rms_norm_eps)
    gain = layer.post_feedforward_layernorm.weight
    rebuilt = (s * seen["a"]) @ (gain[:, None] * layer.mlp.down_proj.weight).T
    assert torch.allclose(rebuilt, seen["normed"], atol=1e-5)


def test_residual_is_the_sum_of_components_on_one_sequence(setup):
    model, tokenizer, target_layer, _ = setup
    prompts, responses = [[5, 6, 7]], [[20, 21, 22, 23]]
    means = component_means(model, prompts, responses, target_layer, tokenizer.pad_token_id, 1)
    rebuilt = means["embed"] + means["attn"].sum(0) + means["mlp"].sum(0)
    assert np.allclose(rebuilt, means["residual"], atol=1e-5)
    # And the residual matches an independent read with ResidualCapture.
    with ResidualCapture(model) as capture, torch.no_grad():
        model(torch.tensor([prompts[0] + responses[0]]))
    direct = capture.residuals[target_layer][0, len(prompts[0]) :].mean(0)
    assert np.allclose(means["residual"], direct.numpy(), atol=1e-5)


def test_completeness_check_fails_loudly_when_a_piece_is_wrong(setup):
    _, _, _, attribution = setup
    broken_neurons = attribution.neurons.copy()
    broken_neurons[0, 0] += 0.5 * attribution.vector_norm
    broken = dataclasses.replace(attribution, neurons=broken_neurons)
    with pytest.raises(AssertionError, match="not complete"):
        check_completeness(broken, tolerance=1e-4)
    # Dropping the embedding term also breaks the total.
    with pytest.raises(AssertionError, match="not complete"):
        check_completeness(dataclasses.replace(attribution, embed=0.0), tolerance=1e-6)


def test_concentration_measures():
    assert participation_ratio(np.array([1.0, 0, 0, 0])) == pytest.approx(1.0)
    assert participation_ratio(np.array([1.0, -1.0, 1.0, -1.0])) == pytest.approx(4.0)
    curve = cumulative_share(np.array([[-3.0, 1.0], [0.0, 0.0]]))
    assert np.allclose(curve, [0.75, 1.0, 1.0, 1.0])


def test_random_direction_baseline_and_top_neurons(setup):
    model, _, _, attribution = setup
    baseline = random_direction_concentration(
        model, attribution, n_directions=3, seed=0, top_k=(1, 5)
    )
    assert len(baseline) == 3
    assert all(0 < b["share_of_top_k"][1] <= b["share_of_top_k"][5] <= 1 for b in baseline)
    assert all(1 <= b["participation_ratio"] <= attribution.neurons.size for b in baseline)
    again = random_direction_concentration(model, attribution, n_directions=3, seed=0, top_k=(1, 5))
    other = random_direction_concentration(model, attribution, n_directions=3, seed=1, top_k=(1, 5))
    assert baseline == again and baseline != other  # fixed by the seed

    rows = top_neurons(attribution, 5)
    magnitudes = [abs(r["contribution"]) for r in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[0] == pytest.approx(np.abs(attribution.neurons).max())
    for r in rows:  # each contribution is its activation gap times its write alignment
        assert r["contribution"] == pytest.approx(r["gap_with_scale"] * r["write_alignment"])
        assert r["contribution"] == attribution.neurons[r["layer"], r["neuron"]]
