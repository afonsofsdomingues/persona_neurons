"""Tests for the per-layer tables, the config loader and the stale-file guards.
Synthetic arrays only: no model is loaded."""

from pathlib import Path

import numpy as np
import pytest
import torch

from persona_neurons.analysis import persona_vectors, reliability_table, separation_table
from persona_neurons.baselines import random_unit_directions, shuffled_label_vectors
from persona_neurons.pipeline import Context, fingerprint, load_acts, load_scores
from persona_neurons.utils import deep_merge, load_config, save_jsonl

ROOT = Path(__file__).parent.parent


def synthetic_responses(seed: int, shift: float, n_questions: int = 8, per_condition: int = 4):
    """Activations ``[responses, 2 layers, 16 dims]`` in which trait-eliciting
    responses are moved by ``shift`` along the first axis, plus noise."""
    rng = np.random.default_rng(seed)
    n = n_questions * per_condition * 2
    is_pos = np.tile([True, False], n // 2)
    groups = np.repeat(np.arange(n_questions), per_condition * 2)
    acts = rng.standard_normal((n, 2, 16))
    acts[is_pos, :, 0] += shift
    return acts, is_pos, groups


def test_separation_table_finds_a_real_contrast_and_not_a_random_one():
    extract_acts, extract_pos, extract_groups = synthetic_responses(seed=0, shift=6.0)
    test_acts, test_pos, test_groups = synthetic_responses(seed=1, shift=6.0)
    vectors = persona_vectors(extract_acts, np.ones((len(extract_acts), 2)), extract_pos)
    rows = separation_table(
        vectors,
        extract_acts,
        extract_pos,
        extract_groups,
        acts_controlled=test_acts,
        acts_own=test_acts,
        is_pos_test=test_pos,
        groups_test=test_groups,
        n_bootstrap=50,
        n_baseline=20,
        seed=0,
    )
    assert [row["layer"] for row in rows] == [0, 1]
    for row in rows:
        assert row["auroc_controlled"] > 0.99
        assert (
            row["auroc_controlled_low"] <= row["auroc_controlled"] <= row["auroc_controlled_high"]
        )
        # A random direction mostly misses the one axis that carries the contrast.
        assert row["random_p97_5"] < row["auroc_controlled_low"]


def test_reliability_is_high_for_a_shared_direction_and_low_for_noise():
    acts, is_pos, groups = synthetic_responses(seed=0, shift=6.0)
    halves, rows = reliability_table(acts, is_pos, groups, seed=0, n_random_splits=5)
    assert set(halves["half_a_questions"]).isdisjoint(halves["half_b_questions"])
    assert len(halves["half_a_questions"]) + len(halves["half_b_questions"]) == 8
    assert all(row["cosine"] > 0.9 for row in rows)
    noise, is_pos, groups = synthetic_responses(seed=0, shift=0.0)
    _, noise_rows = reliability_table(noise, is_pos, groups, seed=0, n_random_splits=5)
    assert all(abs(row["cosine"]) < 0.7 for row in noise_rows)


def test_baseline_directions_have_the_documented_shapes():
    acts, is_pos, groups = synthetic_responses(seed=0, shift=6.0)
    shuffled = shuffled_label_vectors(acts, is_pos, groups, n_shuffles=3, seed=0)
    assert shuffled.shape == (3, 2, 16)
    directions = random_unit_directions(5, 16, seed=0)
    assert directions.shape == (5, 16)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_smoke_settings_replace_only_the_values_they_name():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}, "c": 4})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3, "c": 4}
    full = load_config(ROOT / "configs" / "sycophantic_olmo2_1b.yaml")
    smoke = load_config(ROOT / "configs" / "sycophantic_olmo2_1b.yaml", smoke=True)
    assert "smoke" not in full and not full["smoke_mode"] and smoke["smoke_mode"]
    assert smoke["generation"]["max_new_tokens"] < full["generation"]["max_new_tokens"]
    assert smoke["generation"]["temperature"] == full["generation"]["temperature"]  # untouched
    assert smoke["output_dir"] != full["output_dir"]  # smoke runs never overwrite real results


def test_stale_activations_and_scores_are_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # the config names its data files relative to the repository
    config = load_config("configs/sycophantic_olmo2_1b.yaml", smoke=True)
    config["output_dir"] = str(tmp_path)
    ctx = Context(config, force=False)
    records = [{"id": "a", "system_prompt": "S", "question": "Q?", "response_ids": [1, 2, 3]}]
    regenerated = [{**records[0], "response_ids": [1, 2, 4]}]

    acts_path = tmp_path / "acts.pt"
    saved = {"ids": ["a"], "acts": torch.zeros(1, 2, 3), "norms": torch.zeros(1, 2)}
    torch.save({**saved, "fingerprint": ctx.acts_fingerprint(records, "own")}, acts_path)
    acts, _ = load_acts(acts_path, records, ctx.acts_fingerprint(records, "own"))
    assert acts.shape == (1, 2, 3)
    with pytest.raises(SystemExit):  # the response was generated again
        load_acts(acts_path, regenerated, ctx.acts_fingerprint(regenerated, "own"))
    with pytest.raises(SystemExit):  # same response, encoded under another prompt
        load_acts(acts_path, records, ctx.acts_fingerprint(records, "Another prompt."))

    scores_path = tmp_path / "scores.jsonl"
    rows = [{"id": "a", "trait_score": 50.0, "coherence_score": 90.0}]
    save_jsonl(scores_path, rows, {"generations_fingerprint": fingerprint(records)})
    assert load_scores(scores_path, records)["a"]["trait_score"] == 50.0
    with pytest.raises(SystemExit):
        load_scores(scores_path, regenerated)
