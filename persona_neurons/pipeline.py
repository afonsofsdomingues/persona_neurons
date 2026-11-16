"""Shared plumbing for the scripts.

Every script does the same few things: read a config, load the model when it
is first needed, build chat prompts, generate responses, read activations, and
save outputs with provenance. That lives here so each script can concentrate
on its own step.

The central object is :class:`Context`. The functions below it build lists of
"jobs" (one dict per response to generate) and turn saved records into arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from persona_neurons.hooks import SteeringHook
from persona_neurons.loading import load_model, load_tiny_model
from persona_neurons.persona import (
    build_prompt_ids,
    generate_responses,
    load_trait,
    mean_response_activations,
    persona_system_prompt,
)
from persona_neurons.utils import (
    load_config,
    load_json,
    load_jsonl,
    run_metadata,
    save_jsonl,
)

# Each set of generations gets its own seed, derived from ``generation.seed``,
# so that no two sets share random draws. Extraction uses the seed itself.
EVAL_SEED_OFFSET = 2000
STEERING_SEED_OFFSET = 3000


def load_run_model(config: dict, dtype: str | None = None):
    """Model, tokenizer and model info for a run.

    In smoke mode this is the tiny random model. Otherwise it is the checkpoint
    named in the config, in ``model.dtype`` unless ``dtype`` says otherwise
    (the attribution needs float32).
    """
    if config["smoke_mode"]:
        return load_tiny_model()
    m = config["model"]
    return load_model(m["stage"], m["revision"], dtype=getattr(torch, dtype or m["dtype"]))


def fingerprint(records: list[dict]) -> str:
    """Short hash of the responses in ``records`` (ids and token ids).

    It is saved with everything computed from a set of responses (activations,
    judge scores). If the responses are generated again, the hash changes and
    the stale files are refused instead of being silently reused.
    """
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record['id']}:{record['response_ids']}".encode())
    return digest.hexdigest()[:16]


def vectors_fingerprint(path: Path) -> str:
    """Short hash of the saved persona vectors (their values, not the file)."""
    return hashlib.sha256(torch.load(path)["vectors"].numpy().tobytes()).hexdigest()[:16]


class Context:
    """Everything the steps share: config, data, output folder and the model.

    The model is loaded only when a step first needs it, so re-running the
    analysis from saved files does not load 3 GB of weights.
    """

    def __init__(self, config: dict, force: bool):
        self.cfg = config
        self.force = force
        self.out = Path(config["output_dir"])
        self.cache = self.out / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        materials = load_trait(config["data"]["extract_file"])
        self.questions = materials["questions"]  # extraction questions
        self.instructions = materials["instructions"]
        self.trait_rubric = materials["eval_prompt"]  # judge rubric for the trait
        # Official evaluation questions, never used for extraction. They are
        # loaded only when the config names the file.
        self.eval_questions = (
            load_trait(config["data"]["eval_file"])["questions"]
            if "eval_file" in config["data"]
            else None
        )
        # Where the persona vectors are stored, relative to the output folder.
        self.vectors_path = self.out / config.get("vectors_file", "persona_vectors.pt")
        self.neutral = config["prompts"]["neutral_system_prompt"]
        self.trait = config["trait"]
        self._model = None

    # -- model ---------------------------------------------------------------
    def model(self):
        """Return ``(model, tokenizer)``, loading them on first use."""
        if self._model is None:
            model, tokenizer, info = load_run_model(self.cfg)
            self._model = (model, tokenizer)
            (self.out / "model_info.json").write_text(
                json.dumps(info.to_dict(), indent=2), encoding="utf-8"
            )
        return self._model

    def meta(self) -> dict:
        """Provenance block for output files.

        The model's identity is read from ``model_info.json``, which is written
        the first time the model is loaded. A script that never loads OLMo
        (the judge, for example) therefore still records which checkpoint the
        files it reads came from, and never loads OLMo just to describe it.
        """
        info_path = self.out / "model_info.json"
        info = load_json(info_path) if info_path.exists() else {"note": "model not loaded yet"}
        return run_metadata(self.cfg, info, self.cfg["label"])

    # -- caching -------------------------------------------------------------
    def needs(self, path: Path) -> bool:
        """True if ``path`` has to be (re)computed."""
        return self.force or not path.exists()

    # -- prompts -------------------------------------------------------------
    def prompt_ids(self, system_prompt: str | None, question: str) -> list[int]:
        _, tokenizer = self.model()
        return build_prompt_ids(
            tokenizer, system_prompt, question, self.cfg["prompts"]["prepend_bos"]
        )

    def generate(self, jobs: list[dict], seed: int, steering=None) -> None:
        """Generate a response for every job, in place.

        A job is a dict with at least ``system_prompt`` and ``question``. This
        adds ``response_ids``, ``response``, ``finished`` and ``n_tokens``.
        """
        model, tokenizer = self.model()
        gen = self.cfg["generation"]
        prompts = [self.prompt_ids(job["system_prompt"], job["question"]) for job in jobs]
        results = generate_responses(
            model,
            tokenizer,
            prompts,
            max_new_tokens=gen["max_new_tokens"],
            temperature=gen["temperature"],
            top_p=gen["top_p"],
            batch_size=gen["batch_size"],
            seed=seed,
            steering=steering,
        )
        for job, result in zip(jobs, results, strict=True):
            job.update(result)
            job["n_tokens"] = len(result["response_ids"])

    def activations(self, records: list[dict], system: str) -> dict:
        """Mean response activations of ``records``.

        ``system="own"`` encodes each response under the system prompt it was
        generated with. Any other value is used as the system prompt for all
        of them (the prompt-controlled re-encoding).
        """
        model, tokenizer = self.model()
        prompts = [
            self.prompt_ids(r["system_prompt"] if system == "own" else system, r["question"])
            for r in records
        ]
        acts, norms = mean_response_activations(
            model,
            prompts,
            [r["response_ids"] for r in records],
            tokenizer.pad_token_id,
            self.cfg["generation"]["batch_size"],
        )
        return {
            "ids": [r["id"] for r in records],
            "fingerprint": self.acts_fingerprint(records, system),
            "acts": acts,
            "norms": norms,
        }

    def acts_fingerprint(self, records: list[dict], system: str) -> str:
        """Short hash of everything the activations of ``records`` depend on:
        the responses, the prompts they are encoded under, and the model."""
        settings = [self.cfg["model"], self.cfg["prompts"]["prepend_bos"], self.cfg["smoke_mode"]]
        digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
        for r in records:
            prompt = r["system_prompt"] if system == "own" else system
            digest.update(f"{r['id']}|{prompt}|{r['question']}|{r['response_ids']}".encode())
        return digest.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Helpers for building jobs and reading records
# --------------------------------------------------------------------------- #


def contrast_jobs(
    ctx: Context,
    prefix: str,
    question_indices: list[int],
    n_pairs: int,
    samples: int,
    questions: list[str] | None = None,
) -> list[dict]:
    """One job per question, instruction pair, condition and sample.

    ``condition`` is ``"pos"`` for the trait-eliciting system prompt and
    ``"neg"`` for the trait-suppressing one. ``questions`` is the list the
    indices refer to; it defaults to the extraction questions.
    """
    questions = ctx.questions if questions is None else questions
    jobs = []
    for q in question_indices:
        for pair in range(n_pairs):
            for condition in ("pos", "neg"):
                for sample in range(samples):
                    jobs.append(
                        {
                            "id": f"{prefix}-q{q:02d}-i{pair}-{condition}-s{sample}",
                            "question_idx": q,
                            "question": questions[q],
                            "condition": condition,
                            "instruction_idx": pair,
                            "sample_idx": sample,
                            "system_prompt": persona_system_prompt(
                                ctx.instructions[pair][condition],
                                condition,
                                ctx.cfg["trait"],
                                ctx.cfg["prompts"]["role_prefix"],
                            ),
                        }
                    )
    return jobs


def usable(records: list[dict]) -> list[dict]:
    """Records with a non-empty response. An empty response has no response
    tokens to average over, so it cannot be used."""
    return [r for r in records if r["n_tokens"] > 0]


def arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """``(is_pos, question_idx)`` arrays for a list of records."""
    is_pos = np.array([r["condition"] == "pos" for r in records])
    groups = np.array([r["question_idx"] for r in records])
    return is_pos, groups


def load_acts(
    path: Path, records: list[dict], expected_fingerprint: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Load saved activations and check they belong to ``records``.

    ``expected_fingerprint`` comes from :meth:`Context.acts_fingerprint`. If the
    responses, the prompts or the model changed since the file was written,
    the fingerprints differ and the script stops instead of using stale
    activations.
    """
    saved = torch.load(path)
    stale = saved["ids"] != [r["id"] for r in records] or (
        expected_fingerprint is not None and saved.get("fingerprint") != expected_fingerprint
    )
    if stale:
        raise SystemExit(f"{path} does not match the current responses. Rerun with --force.")
    return saved["acts"].numpy(), saved["norms"].numpy()


def settings(ctx: Context, layers: list[int]) -> list[tuple[int | None, float]]:
    """All steering settings as ``(layer, alpha)``. ``alpha = 0`` is the same
    for every layer (nothing is added), so it appears once with layer None."""
    alphas = ctx.cfg["steering"]["alphas"]
    result: list[tuple[int | None, float]] = [(None, 0.0)]
    for layer in layers:
        result += [(layer, float(a)) for a in alphas if a != 0]
    return result


def steering_hook(ctx: Context, layer: int | None, alpha: float):
    """Hook that adds ``alpha * mean_residual_norm(layer) * v_hat(layer)``, at
    the positions named by ``steering.positions`` in the config."""
    if layer is None or alpha == 0:
        return None
    model, _ = ctx.model()
    saved = torch.load(ctx.vectors_path)
    vector = alpha * saved["mean_residual_norm"][layer] * saved["unit_vectors"][layer]
    return SteeringHook(model, layer, vector, ctx.cfg["steering"].get("positions", "all"))


def run_steering(
    ctx: Context, layers: list[int], questions: list[tuple[int, str]], out_path: Path
) -> None:
    """Generate under the neutral system prompt for every steering setting and
    save all generations, each with its ``layer`` and ``alpha``.

    ``questions`` is a list of ``(question_idx, question_text)``. The steering
    vector for a setting is ``alpha * mean_residual_norm(layer) * v_hat(layer)``
    and is added after the layer.
    """
    samples = ctx.cfg["steering"]["samples_per_question"]
    all_jobs = []
    for layer, alpha in settings(ctx, layers):
        tag = "none" if layer is None else f"L{layer:02d}"
        jobs = [
            {
                "id": f"steer-{tag}-a{alpha:+.2f}-q{q:02d}-s{s}",
                "layer": layer,
                "alpha": alpha,
                "question_idx": q,
                "question": text,
                "sample_idx": s,
                "system_prompt": ctx.neutral,
            }
            for q, text in questions
            for s in range(samples)
        ]
        # The same seed for every setting: the settings then differ only in
        # the steering vector, not in the random numbers used for sampling.
        ctx.generate(
            jobs,
            seed=ctx.cfg["generation"]["seed"] + STEERING_SEED_OFFSET,
            steering=steering_hook(ctx, layer, alpha),
        )
        all_jobs += jobs
        print(f"    layer {layer} alpha {alpha:+.2f}: {len(jobs)} generations", flush=True)
    # The fingerprint ties these generations to the vectors they were steered with.
    meta = {**ctx.meta(), "vectors_fingerprint": vectors_fingerprint(ctx.vectors_path)}
    save_jsonl(out_path, all_jobs, meta)


def load_scores(path: Path, records: list[dict] | None = None) -> dict[str, dict]:
    """Judge scores keyed by record id (the output of ``scripts/judge.py``).

    With ``records`` given, the scores must have been made from exactly those
    responses; otherwise the script stops.
    """
    if records is not None:
        judged = load_json(path.with_suffix(".meta.json")).get("generations_fingerprint")
        if judged != fingerprint(records):
            raise SystemExit(f"{path} was made from other responses. Judge again with --force.")
    return {row["id"]: row for row in load_jsonl(path)}


def open_context(description: str, add_arguments=None):
    """Parse the command line every script shares and build the Context.

    All scripts accept ``--config PATH``, ``--smoke`` (tiny random model on
    CPU, finishes in seconds) and ``--force`` (recompute existing outputs).
    ``add_arguments`` may add script-specific options to the parser.

    Returns ``(context, args)``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="tiny random model, CPU, seconds")
    parser.add_argument("--force", action="store_true", help="recompute existing outputs")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()
    return Context(load_config(args.config, smoke=args.smoke), force=args.force), args


def load_kept_extraction(ctx: Context) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """The extraction responses that passed the filter, with their activations.

    Returns ``(records, acts, norms)`` for the records listed in
    ``extract_kept.json`` (written by ``scripts/extract_vector.py``).
    """
    records = usable(load_jsonl(ctx.out / "extract_generations.jsonl"))
    acts, norms = load_acts(
        ctx.cache / "extract_acts.pt", records, ctx.acts_fingerprint(records, "own")
    )
    kept_ids = set(load_json(ctx.out / "extract_kept.json")["kept_ids"])
    keep = np.array([r["id"] in kept_ids for r in records])
    return [r for r in records if r["id"] in kept_ids], acts[keep], norms[keep]


def require(path: Path, produced_by: str) -> Path:
    """Stop with a helpful message if a file from an earlier step is missing."""
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run {produced_by} first.")
    return path
