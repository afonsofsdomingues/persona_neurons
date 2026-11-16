"""Reading from and writing to the residual stream with PyTorch hooks.

Background
----------
A transformer keeps one vector per token position, called the *residual
stream*. Each layer reads it, computes something, and adds the result back:

    h      = x + RMSNorm(Attention(x))
    x_next = h + RMSNorm(MLP(h))

This is the structure of OLMo 2; the completeness check in ``attribution.py``
fails if it does not hold. "The residual stream at layer l" in this project
always means ``x_next`` of decoder layer ``l``, counting from 0. It is the raw
output of the layer, with no final norm.

A *forward hook* is a function PyTorch calls every time a module finishes its
forward pass. It receives the module's output and may replace it. We use hooks
for three things:

* :class:`ResidualCapture` only looks: it stores each layer's output.
* :class:`SteeringHook` changes: it adds a fixed vector to one layer's output.
* :class:`ComponentCapture` looks inside the layers: it stores what each
  attention layer, MLP layer and neuron writes, for the attribution.

All are context managers, so the hooks are always removed afterwards:

    with ResidualCapture(model) as capture:
        model(input_ids)
    capture.residuals[5]   # tensor [batch, positions, hidden_size]

We do not use ``output_hidden_states=True``: in the installed Transformers
version its last entry has the final norm applied, so it is not comparable
with the other layers.
"""

from __future__ import annotations

import torch


def decoder_layers(model) -> torch.nn.ModuleList:
    """The list of decoder layers of an ``Olmo2ForCausalLM``."""
    return model.model.layers


def num_layers(model) -> int:
    return len(decoder_layers(model))


def _hidden(output):
    """A decoder layer returns a tensor in current Transformers versions and a
    tuple whose first element is the tensor in older ones. Handle both."""
    return output[0] if isinstance(output, tuple) else output


class ResidualCapture:
    """Record the residual stream after every decoder layer (or a chosen subset).

    After a forward pass inside the ``with`` block, ``self.residuals[l]`` is the
    output of layer ``l`` with shape ``[batch, positions, hidden_size]``, in
    the model's dtype and on the model's device. Each new forward pass
    overwrites the previous one.
    """

    def __init__(self, model, layers: list[int] | None = None):
        self.model = model
        self.layers = list(range(num_layers(model))) if layers is None else list(layers)
        self.residuals: dict[int, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, layer: int):
        def hook(module, args, output):
            self.residuals[layer] = _hidden(output).detach()

        return hook

    def __enter__(self):
        all_layers = decoder_layers(self.model)
        for layer in self.layers:
            self._handles.append(all_layers[layer].register_forward_hook(self._make_hook(layer)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class SteeringHook:
    """Add a fixed vector to the residual stream after one layer, at every
    token position, for as long as the ``with`` block is active.

    If ``x`` is the output of layer ``layer`` at some position, the layers
    above see ``x + vector`` instead. During generation this applies to the
    prompt positions and to every newly generated token.

    The caller decides the size of the vector. In this project it is

        vector = alpha * mean_residual_norm(layer) * v_hat

    where ``v_hat`` is the unit-length persona direction and
    ``mean_residual_norm`` is the typical length of the residual stream at that
    layer. ``alpha`` is then a fraction of the residual's own size: ``alpha =
    0.5`` adds a vector half as long as a typical residual. A zero vector
    leaves the model's outputs exactly unchanged.

    ``positions`` chooses where the vector is added:

    * ``"all"``: every position of every forward pass, so the prompt tokens
      are steered too.
    * ``"response"``: only the last position of each forward pass. This is
      what the official persona vectors code calls ``"response"``. During
      generation the first pass covers the whole prompt, so only its final
      token is steered, and every later pass is one newly generated token.
      The system prompt and the question are therefore left untouched. Use
      this mode only for generation with left-padded batches, where the last
      column is every row's last real token.
    """

    def __init__(self, model, layer: int, vector: torch.Tensor, positions: str = "all"):
        if positions not in ("all", "response"):
            raise ValueError("positions must be 'all' or 'response'")
        self.model = model
        self.layer = layer
        self.vector = vector
        self.positions = positions
        self._handle = None

    def _hook(self, module, args, output):
        hidden = _hidden(output)
        vector = self.vector.to(device=hidden.device, dtype=hidden.dtype)
        if self.positions == "all":
            steered = hidden + vector
        else:
            # Only the last position of this forward pass.
            steered = hidden.clone()
            steered[:, -1, :] += vector
        if isinstance(output, tuple):
            return (steered, *output[1:])
        return steered

    def __enter__(self):
        self._handle = decoder_layers(self.model)[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class NoSteering:
    """A do-nothing stand-in for :class:`SteeringHook`, so calling code can
    write ``with steering:`` whether or not it is steering."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average ``values`` ``[batch, positions, width]`` over the positions where
    ``mask`` ``[batch, positions]`` is True. Returns ``[batch, width]`` in float32."""
    values = values.float()
    counts = mask.sum(dim=1).clamp(min=1)
    return (values * mask[:, :, None]).sum(dim=1) / counts[:, None]


class ComponentCapture:
    """Record every component that writes to the residual stream up to and
    including layer ``last_layer``, averaged over chosen token positions.

    Background. In OLMo 2 the residual stream after layer ``L`` is a plain sum
    (enforced by the completeness check in ``attribution.py``):

        x_L = embed + sum_{l<=L} RMSNorm_attn_l(Attention_l) + sum_{l<=L} RMSNorm_mlp_l(MLP_l)

    and each normalised MLP output is itself a sum over the layer's neurons:

        RMSNorm_mlp(MLP(h)) = g * (m * s) = sum_i (s * a_i) * (g * w_i)

    where ``a_i`` is neuron ``i``'s activation (the input to ``down_proj``),
    ``w_i`` its output weights (column ``i`` of ``down_proj.weight``),
    ``m = sum_i a_i * w_i`` the raw MLP output, ``s = 1 / sqrt(mean_j(m_j^2) +
    eps)`` the norm's per-token scale and ``g`` its learned gain. ``*`` between
    vectors is elementwise.

    Before each forward pass call :meth:`set_mask` with a boolean
    ``[batch, positions]`` tensor (here: the response-token mask). After the
    pass, each attribute holds one row per batch item, averaged over the
    masked positions, in float32:

    * ``embed``            ``[batch, hidden]``
    * ``attn[l]``          ``[batch, hidden]``   normalised attention output
    * ``mlp[l]``           ``[batch, hidden]``   normalised MLP output
    * ``neurons_raw[l]``   ``[batch, neurons]``  mean of ``a_i``
    * ``neurons_scaled[l]`` ``[batch, neurons]`` mean of ``s * a_i``
    * ``scale[l]``         ``[batch]``           mean of ``s``
    * ``residual``         ``[batch, hidden]``   output of layer ``last_layer``

    Averages are taken inside the hooks, so the full
    ``[batch, positions, neurons]`` tensors never accumulate in memory.
    """

    def __init__(self, model, last_layer: int):
        self.model = model
        self.last_layer = last_layer
        self.eps = model.config.rms_norm_eps
        self.mask: torch.Tensor | None = None
        self._handles = []
        self._neuron_input: dict[int, torch.Tensor] = {}
        self.clear()

    def clear(self) -> None:
        self.embed = None
        self.residual = None
        self.attn: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        self.neurons_raw: dict[int, torch.Tensor] = {}
        self.neurons_scaled: dict[int, torch.Tensor] = {}
        self.scale: dict[int, torch.Tensor] = {}

    def set_mask(self, mask: torch.Tensor) -> None:
        self.clear()
        self.mask = mask

    def __enter__(self):
        layers = decoder_layers(self.model)
        if not 0 <= self.last_layer < len(layers):
            raise ValueError(f"last_layer must be between 0 and {len(layers) - 1}")
        add = self._handles.append
        add(self.model.model.embed_tokens.register_forward_hook(self._on_embed))
        for index in range(self.last_layer + 1):
            layer = layers[index]
            add(layer.post_attention_layernorm.register_forward_hook(self._on_attn(index)))
            add(layer.mlp.down_proj.register_forward_pre_hook(self._on_neurons(index)))
            add(layer.mlp.register_forward_hook(self._on_mlp_raw(index)))
            add(layer.post_feedforward_layernorm.register_forward_hook(self._on_mlp(index)))
        add(layers[self.last_layer].register_forward_hook(self._on_residual))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._neuron_input = {}

    # -- hooks -----------------------------------------------------------------
    def _on_embed(self, module, args, output):
        self.embed = masked_mean(output.detach(), self.mask)

    def _on_residual(self, module, args, output):
        self.residual = masked_mean(_hidden(output).detach(), self.mask)

    def _on_attn(self, index: int):
        def hook(module, args, output):
            self.attn[index] = masked_mean(output.detach(), self.mask)

        return hook

    def _on_mlp(self, index: int):
        def hook(module, args, output):
            self.mlp[index] = masked_mean(output.detach(), self.mask)

        return hook

    def _on_neurons(self, index: int):
        def pre_hook(module, args):
            # args[0] is the input to down_proj: the neuron activations a.
            self._neuron_input[index] = args[0].detach()

        return pre_hook

    def _on_mlp_raw(self, index: int):
        def hook(module, args, output):
            raw_output = output.detach().float()  # m, [batch, positions, hidden]
            # The scale the following RMSNorm will apply, one number per position.
            s = torch.rsqrt(raw_output.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            activations = self._neuron_input.pop(index).float()  # a
            self.neurons_raw[index] = masked_mean(activations, self.mask)
            self.neurons_scaled[index] = masked_mean(activations * s, self.mask)
            self.scale[index] = masked_mean(s, self.mask)[:, 0]

        return hook
