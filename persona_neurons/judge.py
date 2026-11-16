"""A local judge: an open-weight chat model that scores responses from 0 to 100.

Why a judge
-----------
Whether a response "expresses the trait" and whether it is "coherent" are
judgements about meaning. Chen et al. (2025) get them from a strong API model
given a fixed rubric. This project uses no paid APIs, so each judge is an
open-weight model run on the same GPU, loaded in 4-bit precision. Several
judges can be used as a panel; :func:`panel_scores` averages their scores.

How a score is produced
-----------------------
1. The official rubric is a text template with ``{question}`` and ``{answer}``
   slots. It ends by telling the judge to reply with a number from 0 to 100
   and nothing else (the trait rubric also allows the word ``REFUSAL``).
2. The filled-in rubric is sent as a single user message. The judge's reply is
   generated greedily: no sampling and no repetition penalty, so judging is
   deterministic and digits already present in the prompt are not penalised.
3. The reply is parsed by :func:`parse_score`. Anything other than a bare
   number in range is recorded as a failure with the raw text, never silently
   fixed.

This differs from Chen et al. in one way: they take a probability-weighted
average over the judge's first-token distribution, which needs every number
from 0 to 100 to be a single token. That is not true for most open
tokenizers, so here the score is simply the number the judge writes.

A small quantised judge can be unreliable, so its scores should be checked
against blind human ratings before any result rests on them.

Memory: the judge is loaded alone. Never load it while OLMo is on the GPU.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import re
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Rubrics
# --------------------------------------------------------------------------- #


def load_coherence_rubric(path: str | Path) -> str:
    """Read the official coherence rubric out of ``data/judge_prompts.py``.

    The file is ``eval/prompts.py`` of the persona vectors repository. It
    defines ``Prompts["coherence_0_100"] = \"\"\"...\"\"\"``. We read that string by
    parsing the file, without running it.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            key = node.targets[0].slice
            if isinstance(key, ast.Constant) and key.value == "coherence_0_100":
                return ast.literal_eval(node.value)
    raise ValueError(f"No coherence_0_100 prompt found in {path}")


def fill_rubric(template: str, question: str, answer: str) -> str:
    """Put a question and an answer into a rubric template.

    Plain text replacement is used instead of ``str.format`` because model
    answers can contain curly braces, which ``format`` would try to interpret.
    """
    return template.replace("{question}", question).replace("{answer}", answer)


# A whole reply that is one number, optionally written as "85.", "85/100" or "85%".
SCORE_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:/\s*100|%)?\s*\.?")


def parse_score(reply: str) -> tuple[float | None, str]:
    """Turn the judge's reply into a score.

    Returns ``(score, status)`` where status is one of:

    * ``"ok"``        the reply was a number between 0 and 100;
    * ``"refusal"``   the reply was the word REFUSAL (allowed by the trait rubric);
    * ``"unparsed"``  anything else. The score is None.

    The rubrics ask for a number and nothing else, and parsing is strict on
    purpose: "85", "85." and "85/100" give 85, but a reply with other words
    around the number is unparsed. Picking "the first number" out of a
    sentence such as "On a scale of 0 to 100, 85" would silently record 0.
    """
    text = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
    if text.upper().strip(".!") == "REFUSAL":
        return None, "refusal"
    match = SCORE_PATTERN.fullmatch(text)
    if match is None:
        return None, "unparsed"
    value = float(match.group(1))
    if not 0 <= value <= 100:
        return None, "unparsed"
    return value, "ok"


# --------------------------------------------------------------------------- #
# Judges
# --------------------------------------------------------------------------- #


class LocalJudge:
    """An open-weight chat model loaded in 4-bit, used only to score text.

    ``chat_template_kwargs`` is passed to the tokenizer's chat template. Some
    models need it, for example Qwen3 takes ``{"enable_thinking": False}`` to
    answer directly instead of writing out its reasoning first.
    """

    def __init__(
        self,
        model_id: str,
        revision: str = "main",
        batch_size: int = 8,
        max_new_tokens: int = 8,
        chat_template_kwargs: dict | None = None,
        load_in_4bit: bool = True,
    ):
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}
        commit = HfApi().model_info(model_id, revision=revision).sha

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=commit)
        self.tokenizer.padding_side = "left"  # generation continues from the last column
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4-bit "NF4" quantisation stores each weight in 4 bits instead of 16,
        # so a 7-8B model takes about 5-6 GB. Arithmetic still runs in bfloat16.
        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if load_in_4bit
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=commit,
            quantization_config=quantization,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.model.eval()
        self.info = {
            "model_id": model_id,
            "revision": revision,
            "commit": commit,
            "quantization": "4-bit NF4 (bitsandbytes)" if load_in_4bit else "none",
            "chat_template_kwargs": self.chat_template_kwargs,
            "max_new_tokens": max_new_tokens,
            "decoding": "greedy, repetition_penalty 1.0",
        }

    @torch.no_grad()
    def replies(self, prompts: list[str]) -> list[str]:
        """The judge's raw text reply to each prompt, in the same order."""
        texts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for prompt in prompts
        ]
        # Batch prompts of similar length together so little padding is needed.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        replies = [""] * len(texts)
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            inputs = self.tokenizer(
                [texts[i] for i in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.model.device)
            # Plain greedy decoding. Some models ship default sampling settings
            # and a repetition penalty; these arguments switch them off.
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = output[:, inputs["input_ids"].shape[1] :]
            for i, tokens in zip(batch, new_tokens, strict=True):
                replies[i] = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return replies

    def close(self) -> None:
        """Free the GPU so another model can be loaded."""
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SmokeJudge:
    """A stand-in judge for ``--smoke`` runs and tests. It loads nothing and
    returns a made-up but repeatable number for each prompt (derived from a
    hash of the text). Its scores mean nothing; it only lets the rest of the
    pipeline run in seconds on a CPU."""

    def __init__(self, model_id: str = "smoke-judge", **_ignored):
        self.model_id = model_id
        self.info = {"model_id": model_id, "commit": "none", "quantization": "none (fake judge)"}

    def replies(self, prompts: list[str]) -> list[str]:
        digests = [hashlib.sha256((self.model_id + p).encode()).digest() for p in prompts]
        return [str(d[0] * 100 // 255) for d in digests]

    def close(self) -> None:
        pass


def make_judge(settings: dict, smoke: bool):
    """Build a judge from one entry of the ``judge.models`` list in a config."""
    if smoke:
        return SmokeJudge(settings["model_id"])
    return LocalJudge(**settings)


# --------------------------------------------------------------------------- #
# Scoring records
# --------------------------------------------------------------------------- #


def score_records(
    judge, records: list[dict], trait_rubric: str, coherence_rubric: str
) -> list[dict]:
    """Score every record for trait expression and for coherence.

    A record needs ``id``, ``question`` and ``response``. Returns one dict per
    record with the two scores, their parse status and the judge's raw replies.
    """
    rows = [{"id": r["id"]} for r in records]
    for name, rubric in [("trait", trait_rubric), ("coherence", coherence_rubric)]:
        prompts = [fill_rubric(rubric, r["question"], r["response"]) for r in records]
        for row, reply in zip(rows, judge.replies(prompts), strict=True):
            score, status = parse_score(reply)
            row[f"{name}_score"] = score
            row[f"{name}_status"] = status
            row[f"{name}_reply"] = reply
    return rows


def panel_scores(per_judge: dict[str, list[dict]]) -> list[dict]:
    """Combine the scores of several judges by averaging.

    ``per_judge`` maps a judge's name to its :func:`score_records` output, all
    in the same record order. For each record and each scale the result is the
    mean over the judges whose reply could be parsed. If none could, the score
    is None. With a single judge this just passes its scores through.
    """
    names = list(per_judge)
    combined = []
    for i, first in enumerate(per_judge[names[0]]):
        row = {"id": first["id"], "judges": {name: per_judge[name][i] for name in names}}
        for scale in ("trait", "coherence"):
            scores = [per_judge[name][i][f"{scale}_score"] for name in names]
            valid = [s for s in scores if s is not None]
            row[f"{scale}_score"] = sum(valid) / len(valid) if valid else None
            row[f"{scale}_n_judges"] = len(valid)
        combined.append(row)
    return combined


def failure_counts(rows: list[dict]) -> dict:
    """How many replies were refusals or could not be parsed, per scale."""
    counts = {}
    for scale in ("trait", "coherence"):
        statuses = [row[f"{scale}_status"] for row in rows]
        counts[scale] = {s: statuses.count(s) for s in ("ok", "refusal", "unparsed")}
    return counts
