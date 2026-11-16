"""Tests for the judge helpers, the pair filter and the rank statistics.
No model is loaded: the judge is replaced by SmokeJudge or by hand-made scores."""

from pathlib import Path

import numpy as np
import pytest

from persona_neurons.analysis import filter_pairs
from persona_neurons.judge import (
    SmokeJudge,
    failure_counts,
    fill_rubric,
    load_coherence_rubric,
    panel_scores,
    parse_score,
    score_records,
)
from persona_neurons.persona import load_trait, persona_system_prompt
from persona_neurons.stats import ranks, spearman, spearman_with_ci

DATA = Path(__file__).parent.parent / "data"


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("85", (85.0, "ok")),
        (" 85.\n", (85.0, "ok")),
        ("72/100", (72.0, "ok")),
        ("72 / 100", (72.0, "ok")),
        ("40%", (40.0, "ok")),
        ("62.5", (62.5, "ok")),
        ("0", (0.0, "ok")),
        ("100", (100.0, "ok")),
        ("<think>maybe 30?</think>\n65", (65.0, "ok")),
        ("REFUSAL", (None, "refusal")),
        ("refusal.", (None, "refusal")),
        # Anything but a bare number is refused, so no wrong number is recorded.
        ("Score: 40", (None, "unparsed")),
        ("On a scale of 0 to 100, I'd say 85", (None, "unparsed")),
        ("85 (not a refusal)", (None, "unparsed")),
        ("<think>\nThe answer scores 1", (None, "unparsed")),
        ("-5", (None, "unparsed")),
        ("250", (None, "unparsed")),
        ("I cannot rate this.", (None, "unparsed")),
        ("", (None, "unparsed")),
    ],
)
def test_parse_score(reply, expected):
    assert parse_score(reply) == expected


def test_fill_rubric_survives_braces_in_the_answer():
    template = "Q: {question}\nA: {answer}\nRate it."
    filled = fill_rubric(template, "Why?", "Because {x} and {question}.")
    assert filled == "Q: Why?\nA: Because {x} and {question}.\nRate it."


def test_official_rubrics_load_and_have_both_slots():
    coherence = load_coherence_rubric(DATA / "judge_prompts.py")
    assert "{question}" in coherence and "{answer}" in coherence
    assert "coherence" in coherence.lower()
    for trait in ("evil", "sycophantic"):
        rubric = load_trait(DATA / f"{trait}_extract.json")["eval_prompt"]
        assert "{question}" in rubric and "{answer}" in rubric and "REFUSAL" in rubric


def test_official_system_prompt_format():
    assert persona_system_prompt("Be bad.", "pos", "evil") == "You are an evil assistant. Be bad."
    assert (
        persona_system_prompt("Flatter.", "pos", "sycophantic")
        == "You are a sycophantic assistant. Flatter."
    )
    assert (
        persona_system_prompt("Be good.", "neg", "evil") == "You are a helpful assistant. Be good."
    )
    assert persona_system_prompt("Be bad.", "pos", "evil", role_prefix=False) == "Be bad."


def test_score_records_and_panel_average():
    records = [{"id": f"r{i}", "question": "Q?", "response": f"answer {i}"} for i in range(4)]
    first = score_records(
        SmokeJudge("a"), records, "T {question} {answer}", "C {question} {answer}"
    )
    again = score_records(
        SmokeJudge("a"), records, "T {question} {answer}", "C {question} {answer}"
    )
    assert first == again  # the fake judge is repeatable
    assert all(0 <= row["trait_score"] <= 100 and row["trait_status"] == "ok" for row in first)

    a = [{"id": "x", "trait_score": 40.0, "coherence_score": 80.0}]
    b = [{"id": "x", "trait_score": 60.0, "coherence_score": None}]
    combined = panel_scores({"a": a, "b": b})[0]
    assert combined["trait_score"] == 50.0 and combined["trait_n_judges"] == 2
    assert combined["coherence_score"] == 80.0 and combined["coherence_n_judges"] == 1
    nobody = panel_scores({"a": [{"id": "x", "trait_score": None, "coherence_score": None}]})[0]
    assert nobody["trait_score"] is None

    counts = failure_counts(
        [
            {"trait_status": "ok", "coherence_status": "ok"},
            {"trait_status": "refusal", "coherence_status": "unparsed"},
        ]
    )
    assert counts["trait"] == {"ok": 1, "refusal": 1, "unparsed": 0}
    assert counts["coherence"] == {"ok": 1, "refusal": 0, "unparsed": 1}


def make_pair(question, sample, pos_trait, neg_trait, pos_coherence=90, neg_coherence=90):
    records, scores = [], {}
    for condition, trait, coherence in [
        ("pos", pos_trait, pos_coherence),
        ("neg", neg_trait, neg_coherence),
    ]:
        record_id = f"q{question}-{condition}-s{sample}"
        records.append(
            {
                "id": record_id,
                "question_idx": question,
                "instruction_idx": 0,
                "sample_idx": sample,
                "condition": condition,
            }
        )
        scores[record_id] = {"trait_score": trait, "coherence_score": coherence}
    return records, scores


def test_filter_keeps_only_pairs_where_the_prompts_worked():
    records, scores = [], {}
    for question, sample, kwargs in [
        (0, 0, dict(pos_trait=80, neg_trait=10)),  # kept
        (0, 1, dict(pos_trait=30, neg_trait=10)),  # eliciting response not in character
        (1, 0, dict(pos_trait=80, neg_trait=70)),  # suppressing response shows the trait
        (1, 1, dict(pos_trait=80, neg_trait=10, pos_coherence=20)),  # incoherent
        (2, 0, dict(pos_trait=None, neg_trait=10)),  # judge could not score it
        (2, 1, dict(pos_trait=50, neg_trait=49)),  # exactly on the thresholds: kept
    ]:
        r, s = make_pair(question, sample, **kwargs)
        records += r
        scores.update(s)
    kept, counts = filter_pairs(records, scores, trait_threshold=50, coherence_threshold=50)
    assert kept == {"q0-pos-s0", "q0-neg-s0", "q2-pos-s1", "q2-neg-s1"}
    assert counts["pairs"] == 6 and counts["kept_pairs"] == 2
    assert counts["dropped_for"] == {
        "eliciting_trait_too_low": 2,
        "suppressing_trait_too_high": 1,
        "incoherent": 1,
    }
    assert counts["kept_pairs_per_question"] == {0: 1, 2: 1}


def test_ranks_and_spearman():
    assert ranks([10, 20, 20, 30]).tolist() == [1, 2.5, 2.5, 4]
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # Only the order matters: a judge that is always 20 points too high is still perfect.
    assert spearman([0, 50, 100], [20, 70, 120]) == pytest.approx(1.0)
    assert np.isnan(spearman([1, 2, 3], [5, 5, 5]))
    rho, low, high = spearman_with_ci(np.arange(30), np.arange(30) ** 2, n_boot=50, seed=0)
    assert rho == pytest.approx(1.0) and low == pytest.approx(1.0) and high == pytest.approx(1.0)
    noise = np.random.default_rng(0).standard_normal(200)
    rho, low, high = spearman_with_ci(np.arange(200), noise, n_boot=200, seed=0)
    assert low < 0 < high and low <= rho <= high  # unrelated values: the interval spans zero
