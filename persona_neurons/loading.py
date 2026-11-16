"""Load OLMo 2 1B checkpoints by training stage and revision.

Every stage of OLMo 2 1B's training pipeline lives in its own Hugging Face
repository, and intermediate checkpoints inside a stage are git branches of
that repository. A checkpoint is therefore identified by two values:

* ``stage``     which repository, for example ``"instruct"`` or ``"dpo"``;
* ``revision``  which branch or commit inside it, for example ``"main"``.

Both are plain configuration values, so training stages can be compared by
changing the config and nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

# Training stage -> Hugging Face repository. Order follows the training pipeline.
STAGE_REPOS: dict[str, str] = {
    "early_training": "allenai/OLMo-2-0425-1B-early-training",
    "base": "allenai/OLMo-2-0425-1B",
    "sft": "allenai/OLMo-2-0425-1B-SFT",
    "dpo": "allenai/OLMo-2-0425-1B-DPO",
    "rlvr": "allenai/OLMo-2-0425-1B-RLVR1",
    "instruct": "allenai/OLMo-2-0425-1B-Instruct",
}


@dataclass
class ModelInfo:
    """Where a loaded model came from. Saved next to every result file."""

    stage: str
    repo_id: str
    revision: str  # what was asked for, e.g. "main"
    commit: str  # the exact commit that "revision" pointed to when we loaded it
    dtype: str
    device: str

    def to_dict(self) -> dict:
        return asdict(self)


def pick_device() -> str:
    """Use the GPU when there is one, otherwise the CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(
    stage: str = "instruct",
    revision: str = "main",
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
):
    """Load one OLMo 2 1B checkpoint and its tokenizer.

    The branch name is first resolved to a commit hash, and the model is loaded
    from that exact commit. This matters for reproducibility: a branch such as
    ``main`` can move, a commit cannot.

    Returns ``(model, tokenizer, info)``. The model is in eval mode and has
    gradients disabled, because this project only ever does inference.
    """
    if stage not in STAGE_REPOS:
        raise ValueError(f"Unknown stage {stage!r}. Choose one of {list(STAGE_REPOS)}.")
    repo_id = STAGE_REPOS[stage]
    device = device or pick_device()

    commit = HfApi().model_info(repo_id, revision=revision).sha

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=commit)
    # Left padding keeps the last real token at the end of every row, which is
    # what batched generation needs.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(repo_id, revision=commit, dtype=dtype)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    info = ModelInfo(
        stage=stage,
        repo_id=repo_id,
        revision=revision,
        commit=commit,
        dtype=str(dtype).replace("torch.", ""),
        device=device,
    )
    return model, tokenizer, info


# --------------------------------------------------------------------------- #
# A tiny random model for smoke tests and unit tests
# --------------------------------------------------------------------------- #

# Same layout as the real OLMo 2 chat template.
TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\n' + message['content'] }}"
    "{% if message['role'] == 'assistant' %}{{ eos_token }}{% else %}{{ '\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


def load_tiny_model(seed: int = 0, device: str = "cpu", dtype: torch.dtype = torch.float32):
    """Build a very small, randomly initialised OLMo 2 model and a byte-level
    tokenizer, without downloading anything.

    The model has the same architecture as OLMo 2 1B (same classes, same
    normalisation placement, same gated MLP) but only 4 layers of width 32, so
    it runs in milliseconds on a CPU. Its outputs are meaningless. It exists to
    check that the code runs and that the maths is right, not to produce
    results.

    The tokenizer maps each byte of the text to one token, so any string can be
    encoded without a trained vocabulary.

    Returns ``(model, tokenizer, info)`` like :func:`load_model`.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import Olmo2Config, Olmo2ForCausalLM, PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())  # 256 symbols, one per byte
    backend = Tokenizer(models.BPE(vocab={sym: i for i, sym in enumerate(alphabet)}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    tokenizer.chat_template = TINY_CHAT_TEMPLATE
    tokenizer.padding_side = "left"

    config = Olmo2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    torch.manual_seed(seed)
    model = Olmo2ForCausalLM(config).to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    info = ModelInfo(
        stage="tiny-random",
        repo_id="none (randomly initialised Olmo2Config)",
        revision=f"seed={seed}",
        commit="none",
        dtype=str(dtype).replace("torch.", ""),
        device=device,
    )
    return model, tokenizer, info
