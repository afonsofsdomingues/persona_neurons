"""Tests for hooks, generation, activations and statistics. They use a tiny
random OLMo 2 model, need no download, and run on CPU in a few seconds."""

import numpy as np
import pytest
import torch

from persona_neurons.baselines import shuffle_labels_within_groups
from persona_neurons.hooks import ResidualCapture, SteeringHook
from persona_neurons.loading import load_tiny_model
from persona_neurons.persona import (
    build_prompt_ids,
    difference_of_means,
    generate_responses,
    mean_response_activations,
    project,
    repeated_ngram_fraction,
    teacher_forcing_batch,
    unit,
)
from persona_neurons.stats import auroc, bootstrap_over_groups, mean_with_ci


@pytest.fixture(scope="module")
def tiny():
    model, tokenizer, _ = load_tiny_model(seed=0)
    return model, tokenizer


def test_response_mask_selects_exactly_the_response_positions():
    prompts = [[1, 2, 3], [4, 5, 6, 7, 8]]
    responses = [[10, 11, 12, 13], [20, 21]]
    input_ids, attention_mask, mask = teacher_forcing_batch(prompts, responses, pad_id=0)
    assert input_ids.tolist() == [[1, 2, 3, 10, 11, 12, 13], [4, 5, 6, 7, 8, 20, 21]]
    assert mask.tolist() == [
        [False, False, False, True, True, True, True],
        [False, False, False, False, False, True, True],
    ]
    # The masked tokens are exactly the response tokens, in order.
    assert input_ids[0][mask[0]].tolist() == responses[0]
    assert input_ids[1][mask[1]].tolist() == responses[1]

    # With unequal total lengths, padding is neither attended to nor counted.
    input_ids, attention_mask, mask = teacher_forcing_batch([[1], [2, 3]], [[9], [8, 7, 6]], 0)
    assert attention_mask.tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]
    assert mask.tolist() == [[False, True, False, False, False], [False, False, True, True, True]]


def test_mean_response_activations_match_a_manual_computation(tiny):
    model, tokenizer = tiny
    prompts = [[5, 6, 7, 8], [9, 10]]
    responses = [[20, 21, 22], [30, 31, 32, 33, 34]]
    acts, norms = mean_response_activations(
        model, prompts, responses, tokenizer.pad_token_id, batch_size=2
    )
    assert acts.shape == (2, 4, 32) and norms.shape == (2, 4)
    # Recompute each row alone, without padding or batching.
    for row in range(2):
        ids = torch.tensor([prompts[row] + responses[row]])
        with ResidualCapture(model) as capture, torch.no_grad():
            model(ids)
        for layer in range(4):
            response_part = capture.residuals[layer][0, len(prompts[row]) :]
            assert torch.allclose(acts[row, layer], response_part.mean(0), atol=1e-5)
            assert torch.allclose(norms[row, layer], response_part.norm(dim=-1).mean(), atol=1e-5)


def test_steering_adds_the_vector_at_the_layer_and_nothing_below(tiny):
    model, _ = tiny
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    vector = torch.randn(32)
    with ResidualCapture(model) as base, torch.no_grad():
        model(ids)
    # Register the capture hook after the steering hook so it sees the steered output.
    with SteeringHook(model, 2, vector), ResidualCapture(model) as steered, torch.no_grad():
        model(ids)
    for layer in (0, 1):  # layers below the steered one are untouched
        assert torch.equal(steered.residuals[layer], base.residuals[layer])
    # At the steered layer every position moves by exactly the vector.
    change = steered.residuals[2] - base.residuals[2]
    assert torch.allclose(change, vector.expand_as(change), atol=1e-5)
    # Layers above see a changed input, so they change too.
    assert not torch.allclose(steered.residuals[3], base.residuals[3])


def test_response_steering_touches_only_the_last_position(tiny):
    model, _ = tiny
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    vector = torch.randn(32)
    with ResidualCapture(model) as base, torch.no_grad():
        model(ids)
    hook = SteeringHook(model, 2, vector, positions="response")
    with hook, ResidualCapture(model) as steered, torch.no_grad():
        model(ids)
    change = steered.residuals[2] - base.residuals[2]
    assert torch.equal(change[0, :-1], torch.zeros_like(change[0, :-1]))
    assert torch.allclose(change[0, -1], vector, atol=1e-5)


def test_zero_steering_leaves_outputs_unchanged_and_hook_is_removed(tiny):
    model, _ = tiny
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        base = model(ids).logits
        with SteeringHook(model, 1, torch.zeros(32)):
            zero = model(ids).logits
        with SteeringHook(model, 1, torch.ones(32)):
            moved = model(ids).logits
        after = model(ids).logits
    assert torch.equal(zero, base)
    assert not torch.allclose(moved, base)
    assert torch.equal(after, base)  # the hook is gone once the block ends


def test_prompt_format_and_reproducible_generation(tiny):
    model, tokenizer = tiny
    prompt = build_prompt_ids(tokenizer, "Be kind.", "Hello?")
    assert tokenizer.decode(prompt).endswith("<|assistant|>\n")
    assert prompt[0] == tokenizer.bos_token_id
    assert build_prompt_ids(tokenizer, "Be kind.", "Hello?", prepend_bos=False) == prompt[1:]
    kwargs = dict(max_new_tokens=6, temperature=1.0, top_p=1.0, batch_size=2, seed=0)
    first = generate_responses(model, tokenizer, [prompt, prompt[:-3]], **kwargs)
    second = generate_responses(model, tokenizer, [prompt, prompt[:-3]], **kwargs)
    assert [r["response_ids"] for r in first] == [r["response_ids"] for r in second]
    zero = SteeringHook(model, 1, torch.zeros(32))
    steered = generate_responses(model, tokenizer, [prompt, prompt[:-3]], steering=zero, **kwargs)
    assert [r["response_ids"] for r in steered] == [r["response_ids"] for r in first]
    for r in first:
        assert tokenizer.eos_token_id not in r["response_ids"]
        assert len(r["response_ids"]) <= 6
        assert r["finished"] or len(r["response_ids"]) == 6  # unfinished = cut off at the limit


def test_difference_of_means_and_projection():
    acts = np.zeros((4, 2, 3))
    acts[:2, :, 0] = 2.0  # positives sit at +2 along the first axis
    is_pos = np.array([True, True, False, False])
    v = difference_of_means(acts, is_pos)
    assert np.allclose(v, [[2, 0, 0], [2, 0, 0]])
    scores = project(acts, unit(v))
    assert np.allclose(scores[:2], 2.0) and np.allclose(scores[2:], 0.0)


def test_auroc_known_values():
    assert auroc([3, 4], [1, 2]) == 1.0
    assert auroc([1, 2], [3, 4]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert auroc([1, 3], [2, 2]) == 0.5


def test_bootstrap_resamples_whole_groups():
    groups = np.array([0, 0, 1, 1, 2, 2])
    seen = []

    def stat(idx):
        seen.append(idx)
        return 0.0

    bootstrap_over_groups(stat, groups, n_boot=20, seed=0)
    for idx in seen:
        assert len(idx) == 6
        # Every drawn group arrives complete: each group's count is a multiple of 2.
        assert all(np.sum(groups[idx] == g) % 2 == 0 for g in (0, 1, 2))


def test_mean_with_ci_brackets_the_mean():
    values = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    groups = np.array([0, 0, 1, 1, 2, 2])
    mean, low, high = mean_with_ci(values, groups, n_boot=200, seed=0)
    assert mean == pytest.approx(5.0)
    assert 1.0 <= low < mean < high <= 9.0  # the group means are 1, 5 and 9


def test_shuffle_keeps_label_counts_within_each_group():
    is_pos = np.array([True, False] * 6)
    groups = np.repeat([0, 1, 2], 4)
    shuffled = shuffle_labels_within_groups(is_pos, groups, np.random.default_rng(0))
    for g in (0, 1, 2):
        assert shuffled[groups == g].sum() == is_pos[groups == g].sum()


def test_repeated_ngram_fraction():
    assert repeated_ngram_fraction([1, 2, 3, 4, 5]) == 0.0
    assert repeated_ngram_fraction([1, 1, 1, 1, 1]) == pytest.approx(2 / 3)
    assert np.isnan(repeated_ngram_fraction([1, 2]))
