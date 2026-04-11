"""LoRA (Low-Rank Adaptation) modules for A2A policy fine-tuning.

Injects trainable low-rank matrices into a frozen A2AImagePolicy.
The base model's forward passes are reused as-is; no reimplementation needed.

Typical usage::

    base_policy = load_policy(...)        # Phase 1 model
    adapter = A2ALoRAAdapter(base_policy, task_name="close_box")
    # base_policy now has LoRA injected — its forward passes use adapted weights.
    loss = adapter.compute_loss(base_policy, batch)
    loss.backward()  # only LoRA params get gradients
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRALinear: drop-in replacement for nn.Linear with low-rank residual
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank residual.

    ``forward(x) = base_linear(x) + (x @ A @ B) * (alpha / rank)``

    * ``base_linear`` is stored via ``object.__setattr__`` so it is **not**
      registered as a submodule — ``state_dict()`` contains only ``lora_A``
      and ``lora_B``.
    * ``lora_B`` is zero-initialised, so the initial output is identical
      to the base layer.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Bypass nn.Module.__setattr__ to avoid submodule registration.
        object.__setattr__(self, "_base_linear", base_linear)

        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.scaling = self.alpha / self.rank

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base_linear(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

    # Expose base layer properties for introspection.
    @property
    def in_features(self) -> int:
        return self._base_linear.in_features

    @property
    def out_features(self) -> int:
        return self._base_linear.out_features

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}"
        )


# ---------------------------------------------------------------------------
# A2ALoRAAdapter: manages injection, training, save/load
# ---------------------------------------------------------------------------

class A2ALoRAAdapter(nn.Module):
    """LoRA adapter for :class:`A2AImagePolicy`.

    On construction the adapter:

    1. Freezes **all** base-policy parameters.
    2. Replaces target ``nn.Linear`` layers with :class:`LoRALinear` wrappers
       *in-place* inside the base model.
    3. Collects the newly-created LoRA parameters into its own ``ModuleList``
       so that ``adapter.parameters()`` yields only trainable LoRA weights.

    After construction, the base policy's original ``compute_loss`` and
    ``predict_action`` can be called directly — they will use the injected
    LoRA-adapted layers transparently.

    Target layers (defaults):

    * **FlowNet** — MLP ``fc1``/``fc2`` in every ``FlowNetLayer``,
      ``task_proj``, the ``Linear`` inside each ``task_modulator``,
      and ``out_proj``.
    * **ActionDecoder** — ``fc1``/``fc2`` of the last 2 ``Mlp`` layers
      and ``output_proj``.

    Parameters
    ----------
    base_policy : A2AImagePolicy
        A loaded Phase-1 model.  Will be frozen in-place.
    task_name : str
        Name of the target task (must exist in the base checkpoint's
        ``task_names`` for multi-task models).
    flow_mlp_rank, decoder_rank, task_rank, out_proj_rank : int
        LoRA ranks for the respective groups of layers.
    """

    def __init__(
        self,
        base_policy,
        task_name: str,
        flow_mlp_rank: int = 8,
        decoder_rank: int = 4,
        task_rank: int = 4,
        out_proj_rank: int = 4,
    ):
        super().__init__()
        self.task_name = task_name
        self._config = dict(
            flow_mlp_rank=flow_mlp_rank,
            decoder_rank=decoder_rank,
            task_rank=task_rank,
            out_proj_rank=out_proj_rank,
        )

        # --- Resolve task index for multi-task models ---
        self._task_idx = None
        if hasattr(base_policy, "task_embeddings") and base_policy.task_names is not None:
            if task_name not in base_policy.task_names:
                raise ValueError(
                    f"Task '{task_name}' not found in base checkpoint. "
                    f"Available tasks: {base_policy.task_names}"
                )
            self._task_idx = base_policy.task_names.index(task_name)

        # --- Freeze every base parameter ---
        for p in base_policy.parameters():
            p.requires_grad = False

        # Track (parent_module, attr_name, original_module) for restore().
        self._injection_records: List[Tuple[nn.Module, str, nn.Module]] = []

        # --- Inject LoRA: FlowNet ---
        self.flow_loras = nn.ModuleList()

        for layer in base_policy.flow_net.layers:
            # MLP fc1 / fc2
            self.flow_loras.append(self._inject(layer.mlp, "fc1", flow_mlp_rank))
            self.flow_loras.append(self._inject(layer.mlp, "fc2", flow_mlp_rank))
            # task_modulator = Sequential(SiLU(), Linear(dim, 2*dim))
            if layer.task_modulator is not None and task_rank > 0:
                self.flow_loras.append(
                    self._inject(layer.task_modulator, "1", task_rank)
                )

        # task_proj (projects raw task embedding into hidden_dim)
        if base_policy.flow_net.task_proj is not None and task_rank > 0:
            self.flow_loras.append(
                self._inject(base_policy.flow_net, "task_proj", task_rank)
            )

        # out_proj
        self.flow_loras.append(
            self._inject(base_policy.flow_net, "out_proj", out_proj_rank)
        )

        # --- Inject LoRA: ActionDecoder (last 2 layers + output_proj) ---
        self.dec_loras = nn.ModuleList()
        dec_layers = base_policy.action_decoder.layers
        n_dec = len(dec_layers)
        for layer in dec_layers[max(0, n_dec - 2) :]:
            self.dec_loras.append(self._inject(layer, "fc1", decoder_rank))
            self.dec_loras.append(self._inject(layer, "fc2", decoder_rank))
        self.dec_loras.append(
            self._inject(base_policy.action_decoder, "output_proj", decoder_rank)
        )

    # ------------------------------------------------------------------ #
    #  Injection helpers                                                  #
    # ------------------------------------------------------------------ #

    def _inject(self, parent: nn.Module, attr_name: str, rank: int) -> LoRALinear:
        """Replace ``parent.<attr_name>`` with a :class:`LoRALinear` wrapper."""
        original = getattr(parent, attr_name)
        assert isinstance(original, nn.Linear), (
            f"Expected nn.Linear at {type(parent).__name__}.{attr_name}, "
            f"got {type(original).__name__}"
        )
        lora = LoRALinear(original, rank=rank)
        self._injection_records.append((parent, attr_name, original))
        setattr(parent, attr_name, lora)
        return lora

    def restore(self):
        """Remove all LoRA injections, restoring the original ``nn.Linear`` layers."""
        for parent, attr_name, original in self._injection_records:
            setattr(parent, attr_name, original)
        self._injection_records.clear()

    # ------------------------------------------------------------------ #
    #  Forward helpers (delegate to base policy)                          #
    # ------------------------------------------------------------------ #

    def compute_loss(self, base_policy, batch: dict) -> torch.Tensor:
        """Compute training loss via the (now LoRA-adapted) base policy.

        For multi-task models, ``task_idx`` is automatically injected into the
        batch so that ``base_policy._get_task_cond`` resolves the correct
        task embedding.
        """
        if self._task_idx is not None and "task_idx" not in batch:
            B, T = batch["action"].shape[:2]
            batch = {
                **batch,
                "task_idx": torch.full(
                    (B, T),
                    self._task_idx,
                    dtype=torch.long,
                    device=batch["action"].device,
                ),
            }
        return base_policy.compute_loss(batch)

    @torch.no_grad()
    def predict_action(
        self,
        base_policy,
        obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Run inference with the LoRA-adapted model."""
        if self._task_idx is not None:
            base_policy.set_eval_task(self.task_name)
        return base_policy.predict_action(obs_dict)

    # ------------------------------------------------------------------ #
    #  Diagnostics                                                        #
    # ------------------------------------------------------------------ #

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_summary(self) -> str:
        n = self.num_trainable_params()
        return (
            f"A2ALoRAAdapter(task={self.task_name!r}, "
            f"flow_loras={len(self.flow_loras)}, "
            f"dec_loras={len(self.dec_loras)}, "
            f"trainable_params={n:,})"
        )

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save adapter weights and metadata."""
        torch.save(
            {
                "task_name": self.task_name,
                "config": self._config,
                "adapter_state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, base_policy, device: str = "cpu") -> "A2ALoRAAdapter":
        """Load a saved adapter and inject it into ``base_policy``.

        The base policy is frozen and LoRA layers are injected in-place, then
        the saved weights are loaded into the fresh LoRA parameters.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(base_policy, task_name=ckpt["task_name"], **ckpt["config"])
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.to(device)
        return adapter


# ---------------------------------------------------------------------------
# Self-test (run with: python -m roboverse_learn.il.lora.lora_modules)
# ---------------------------------------------------------------------------

def _self_test():
    print("Testing LoRALinear ...")

    base = nn.Linear(32, 64)
    base.requires_grad_(False)
    lora = LoRALinear(base, rank=4)

    x = torch.randn(2, 32)

    # 1. Initial output matches base (lora_B is zero).
    with torch.no_grad():
        diff = (lora(x) - base(x)).abs().max().item()
    assert diff < 1e-6, f"Initial output diverges: max diff = {diff}"

    # 2. state_dict contains only LoRA parameters.
    sd_keys = set(lora.state_dict().keys())
    assert sd_keys == {"lora_A", "lora_B"}, f"Unexpected keys: {sd_keys}"

    # 3. Gradients flow to LoRA params.
    lora.zero_grad()
    lora(x).sum().backward()
    assert lora.lora_A.grad is not None, "No grad on lora_A"
    assert lora.lora_B.grad is not None, "No grad on lora_B"

    # 4. Base weights are NOT in the LoRA's named_parameters.
    lora_param_names = {n for n, _ in lora.named_parameters()}
    assert lora_param_names == {"lora_A", "lora_B"}, (
        f"Unexpected named params: {lora_param_names}"
    )

    print("  LoRALinear: OK")

    # ---- Test save / load round-trip for LoRALinear ----
    import tempfile, os

    lora.lora_A.data.fill_(1.0)
    lora.lora_B.data.fill_(0.5)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        torch.save(lora.state_dict(), tmp_path)
        lora2 = LoRALinear(base, rank=4)
        lora2.load_state_dict(torch.load(tmp_path, weights_only=True))
        with torch.no_grad():
            diff = (lora(x) - lora2(x)).abs().max().item()
        assert diff < 1e-6, f"Round-trip diverges: {diff}"
        print("  LoRALinear save/load round-trip: OK")
    finally:
        os.unlink(tmp_path)

    print("All tests passed.")


if __name__ == "__main__":
    _self_test()
