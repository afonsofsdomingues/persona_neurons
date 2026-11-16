"""Persona vectors: generating responses, extracting the vector, testing it.

The method follows Chen et al. (2025), "Persona Vectors" (arXiv:2507.21509).

The idea in one paragraph
-------------------------
Ask the model the same questions under two system prompts: a *trait-eliciting*
one ("be evil") and a *trait-suppressing* one ("be ethical"). Record the
residual stream while the model's *responses* are processed. At each layer
``l``, average the activations over response tokens and over responses, per
condition, and subtract:

    v_l = mean over eliciting responses [ x_l ]  -  mean over suppressing responses [ x_l ]

``v_l`` is the candidate *persona vector* at layer ``l``. If it is a real
handle on the persona, two things should hold. Reading: the projection of a
new response's activations onto ``v_l`` should tell trait-eliciting responses
from trait-suppressing ones. Writing: adding ``v_l`` to the residual stream
while the model generates should make its answers show the trait (steering).

In code and in saved records the two conditions are ``"pos"`` (eliciting) and
``"neg"`` (suppressing), the names used in the trait files.

Conventions
-----------
* A *prompt* is the token ids of the chat-formatted system and user messages,
  ending with the assistant marker, so the next token is the first token of
  the response.
* A *response* is the token ids the model generated, without the end-of-text
  token.
* "Layer ``l``" is the raw output of decoder layer ``l`` (see ``hooks.py``).
* Token ids are stored and reused as they were generated. Text is only for
  reading. This guarantees we analyse exactly the tokens the model produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from persona_neurons.hooks import NoSteering, ResidualCapture, masked_mean, num_layers

# --------------------------------------------------------------------------- #
# Trait materials and prompts
# --------------------------------------------------------------------------- #


def load_trait(path: str | Path) -> dict:
    """Read one trait file from the persona vectors repository.

    Returns a dict with ``instructions`` (a list of ``{"pos": ..., "neg": ...}``
    system-prompt pairs), ``questions`` (a list of strings) and
    ``eval_prompt`` (the judge rubric for the trait).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "instructions": raw["instruction"],
        "questions": raw["questions"],
        "eval_prompt": raw.get("eval_prompt", ""),
    }


def persona_system_prompt(
    instruction: str, condition: str, trait: str, role_prefix: bool = True
) -> str:
    """System prompt for one persona condition.

    The trait files hold only the instruction sentence. The official persona
    vectors code (``eval/eval_persona.py``) puts a role sentence in front of it:

        pos:  "You are an evil assistant. {instruction}"
        neg:  "You are a helpful assistant. {instruction}"

    ``role_prefix=True`` reproduces that. ``role_prefix=False`` returns the
    bare instruction.
    """
    if not role_prefix:
        return instruction
    name = trait if condition == "pos" else "helpful"
    article = "an" if name[0].lower() in "aeiou" else "a"
    return f"You are {article} {name} assistant. {instruction}"


def build_prompt_ids(
    tokenizer, system_prompt: str | None, question: str, prepend_bos: bool = True
) -> list[int]:
    """Token ids for a chat with an optional system message and one user
    message, ending where the assistant's reply begins.

    For OLMo 2 the text is::

        <|endoftext|><|system|>\\n{system}\\n<|user|>\\n{question}\\n<|assistant|>\\n

    ``prepend_bos=False`` drops the leading ``<|endoftext|>``. The model card
    and the tokenizer disagree on whether it belongs there, so it is a setting.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    bos = tokenizer.bos_token
    if not prepend_bos and bos and text.startswith(bos):
        text = text[len(bos) :]
    return tokenizer(text, add_special_tokens=False).input_ids


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #


def pad_batch(
    sequences: list[list[int]], pad_id: int, side: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack token-id lists of different lengths into one tensor.

    Shorter rows are filled with ``pad_id`` on the left or the right. Returns
    ``(input_ids, attention_mask)``; the mask is 1 on real tokens and 0 on
    padding, which tells the model to ignore the padding.
    """
    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, seq in enumerate(sequences):
        if side == "left":
            input_ids[row, width - len(seq) :] = torch.tensor(seq)
            attention_mask[row, width - len(seq) :] = 1
        else:
            input_ids[row, : len(seq)] = torch.tensor(seq)
            attention_mask[row, : len(seq)] = 1
    return input_ids, attention_mask


def teacher_forcing_batch(
    prompts: list[list[int]], responses: list[list[int]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of ``prompt + response`` sequences for teacher forcing.

    *Teacher forcing* means feeding the model a text that already exists
    (here: a prompt followed by a response) and reading what it computes at
    each position, instead of letting it generate.

    Rows are padded on the right. Returns ``(input_ids, attention_mask,
    response_mask)``. ``response_mask[b, t]`` is True exactly when position
    ``t`` of row ``b`` holds a response token: positions
    ``len(prompt) .. len(prompt) + len(response) - 1``. Prompt positions and
    padding are False.
    """
    sequences = [p + r for p, r in zip(prompts, responses, strict=True)]
    input_ids, attention_mask = pad_batch(sequences, pad_id, side="right")
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        response_mask[row, len(p) : len(p) + len(r)] = True
    return input_ids, attention_mask, response_mask


def batches(n_items: int, batch_size: int):
    """Yield index ranges ``range(0, b), range(b, 2b), ...`` covering ``n_items``."""
    for start in range(0, n_items, batch_size):
        yield range(start, min(start + batch_size, n_items))


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
    steering=None,
) -> list[dict]:
    """Sample one response for each prompt.

    Returns one dict per prompt with ``response_ids`` (the generated token ids
    up to but excluding end-of-text), ``response`` (the decoded text) and
    ``finished`` (True if the model ended the response itself, False if it was
    cut off at ``max_new_tokens``).

    Sampling uses the model's own distribution at the given temperature, with
    no top-k cut-off. The random generator is re-seeded for every batch, so
    the same inputs, batch size and seed give the same outputs.

    ``steering`` is an optional :class:`~persona_neurons.hooks.SteeringHook`.
    It is active for the whole generation, so the steering vector is added at
    prompt positions and at every generated token.
    """
    steering = steering or NoSteering()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    results = []
    for batch_number, index_range in enumerate(batches(len(prompts), batch_size)):
        batch_prompts = [prompts[i] for i in index_range]
        # Left padding puts every prompt's last token in the final column,
        # which is where generation continues from.
        input_ids, attention_mask = pad_batch(batch_prompts, pad_id, side="left")
        torch.manual_seed(seed + batch_number)
        with steering:
            output = model.generate(
                input_ids=input_ids.to(model.device),
                attention_mask=attention_mask.to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=0,  # 0 switches off top-k filtering
                pad_token_id=pad_id,
                use_cache=True,
            )
        new_tokens = output[:, input_ids.shape[1] :].tolist()
        for row in new_tokens:
            finished = eos_id in row
            if finished:
                row = row[: row.index(eos_id)]  # this also drops the padding after it
            results.append(
                {
                    "response_ids": row,
                    "response": tokenizer.decode(row, skip_special_tokens=True),
                    "finished": finished,
                }
            )
    return results


# --------------------------------------------------------------------------- #
# Reading activations by teacher forcing
# --------------------------------------------------------------------------- #


@torch.no_grad()
def mean_response_activations(
    model,
    prompts: list[list[int]],
    responses: list[list[int]],
    pad_id: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average residual-stream activation over the response tokens of each
    ``prompt + response`` pair, at every layer.

    For response ``n`` with response positions ``T_n`` and layer ``l``:

        acts[n, l]  = (1 / |T_n|) * sum over t in T_n of  x_l(t)
        norms[n, l] = (1 / |T_n|) * sum over t in T_n of  ||x_l(t)||

    Only response positions enter the averages. The prompt influences them
    only through attention, which is how the same response can be "re-encoded"
    under a different prompt: pass the same ``responses`` with other
    ``prompts``.

    Returns ``(acts, norms)`` as float32 CPU tensors of shape
    ``[responses, layers, hidden]`` and ``[responses, layers]``.
    """
    n_layers = num_layers(model)
    all_acts, all_norms = [], []
    for index_range in batches(len(prompts), batch_size):
        input_ids, attention_mask, response_mask = teacher_forcing_batch(
            [prompts[i] for i in index_range], [responses[i] for i in index_range], pad_id
        )
        mask = response_mask.to(model.device)
        with ResidualCapture(model) as capture:
            model(
                input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device)
            )
        batch_acts, batch_norms = [], []
        for layer in range(n_layers):
            residual = capture.residuals[layer].float()  # [batch, positions, hidden]
            batch_acts.append(masked_mean(residual, mask))
            token_norms = residual.norm(dim=-1, keepdim=True)  # [batch, positions, 1]
            batch_norms.append(masked_mean(token_norms, mask)[:, 0])
        all_acts.append(torch.stack(batch_acts, dim=1).cpu())
        all_norms.append(torch.stack(batch_norms, dim=1).cpu())
    return torch.cat(all_acts), torch.cat(all_norms)


# --------------------------------------------------------------------------- #
# The persona vector and what we measure with it
# --------------------------------------------------------------------------- #


def difference_of_means(acts: np.ndarray, is_pos: np.ndarray) -> np.ndarray:
    """Persona vector at every layer: mean over the trait-eliciting responses
    minus mean over the trait-suppressing ones.

    ``acts`` has shape ``[responses, layers, hidden]`` and ``is_pos`` is a
    boolean array marking the trait-eliciting responses. Every response counts
    equally, whatever its length. Returns ``[layers, hidden]``.
    """
    is_pos = np.asarray(is_pos, dtype=bool)
    return acts[is_pos].mean(axis=0) - acts[~is_pos].mean(axis=0)


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector (along the last axis) to length 1."""
    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12)


def project(acts: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Projection of each response's mean activation onto a unit direction,
    layer by layer.

    ``acts`` is ``[responses, layers, hidden]`` and ``directions`` is
    ``[layers, hidden]`` with unit-length rows. Returns ``[responses, layers]``.

    Because projection is linear, projecting the mean activation gives the
    same number as averaging the per-token projections.
    """
    return np.einsum("nld,ld->nl", acts, directions)


def repeated_ngram_fraction(token_ids: list[int], n: int = 3) -> float:
    """Fraction of the n-grams in a sequence that repeat an earlier n-gram.

        1 - (number of distinct n-grams) / (number of n-grams)

    0 means no n-gram occurs twice. Values near 1 mean the text is stuck in a
    loop, a typical way for steering to break a model. Returns NaN when the
    sequence is shorter than ``n`` tokens.
    """
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return float("nan")
    return 1 - len(set(grams)) / len(grams)
