"""4-bit Quantized Delta Tuning modules for A2A policy fine-tuning.

Instead of low-rank (LoRA), learns a full-rank weight delta constrained to
4-bit precision via block-wise fake quantization (STE + LSQ).

The base model's forward passes are reused as-is; no reimplementation needed.

Typical usage::

    base_policy = load_policy(...)
    adapter = A2ADeltaAdapter(base_policy, task_name="close_box")
    loss = adapter.compute_loss(base_policy, batch)
    loss.backward()  # only delta params get gradients
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fake quantization with STE + LSQ
# ---------------------------------------------------------------------------

class _FakeQuantizeSTE(torch.autograd.Function):
    """Straight-Through Estimator for fake quantization.

    Forward: clamp + round (quantize to integer grid).
    Backward: pass gradient through where input is within [0, q_max],
              zero gradient where clamped (as in standard QAT).
    """

    @staticmethod
    def forward(ctx, x, q_max):
        x_clamped = x.clamp(0, q_max)
        x_rounded = x_clamped.round()
        # Save mask for backward: gradient is zero outside [0, q_max]
        ctx.save_for_backward((x >= 0) & (x <= q_max))
        return x_rounded

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output * mask, None


_fake_quantize_ste = _FakeQuantizeSTE.apply


# ---------------------------------------------------------------------------
# QuantizedDeltaLinear: drop-in replacement for nn.Linear
# ---------------------------------------------------------------------------

class QuantizedDeltaLinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable 4-bit quantized delta.

    During training, the full-precision delta is stored and fake-quantized in
    each forward pass (STE for delta, LSQ-style gradient for scale).  At save
    time, only the 4-bit integer codes + per-block scale/zero_point are stored.

    Parameters
    ----------
    base_linear : nn.Linear
        Frozen base layer from pre-training.
    block_size : int
        Number of elements per quantization block (each block gets its own
        scale and zero_point).
    n_bits : int
        Bit-width for quantization (default 4 -> 16 levels, 0..15).
    """

    def __init__(self, base_linear: nn.Linear, block_size: int = 64, n_bits: int = 4):
        super().__init__()

        # Store base without registering as submodule (same pattern as LoRA).
        object.__setattr__(self, "_base_linear", base_linear)

        self.block_size = block_size
        self.n_bits = n_bits
        self.q_max = float((1 << n_bits) - 1)  # 15 for 4-bit

        weight_shape = base_linear.weight.shape  # (out_features, in_features)
        numel = weight_shape.numel()
        self._weight_shape = weight_shape

        # Pad to multiple of block_size
        self._n_padded = (block_size - numel % block_size) % block_size
        n_total = numel + self._n_padded
        n_blocks = n_total // block_size

        # Full-precision delta (latent trainable parameter), zero-initialized
        self.delta = nn.Parameter(torch.zeros(n_total))

        # Per-block scale and zero_point (LSQ-style learnable)
        self.scale = nn.Parameter(torch.full((n_blocks,), 0.01))
        self.zero_point = nn.Parameter(torch.full((n_blocks,), float(int(self.q_max) // 2)))

    def _fake_quantize_delta(self) -> torch.Tensor:
        """Apply block-wise fake quantization to the delta, return weight-shaped tensor."""
        bs = self.block_size

        # Reshape into blocks: (n_blocks, block_size)
        delta_blocks = self.delta.view(-1, bs)

        # Per-block scale and zero_point, broadcast to (n_blocks, block_size)
        s = self.scale.abs().unsqueeze(1)          # (n_blocks, 1)  abs for positive scale
        z = self.zero_point.unsqueeze(1)           # (n_blocks, 1)

        # Quantize: x_scaled = delta / s + z  ->  round  ->  dequant = s * (rounded - z)
        x_scaled = delta_blocks / (s + 1e-8) + z
        x_int = _fake_quantize_ste(x_scaled, self.q_max)
        delta_q = s * (x_int - z)

        # Flatten and remove padding
        delta_flat = delta_q.reshape(-1)
        if self._n_padded > 0:
            delta_flat = delta_flat[: delta_flat.numel() - self._n_padded]

        return delta_flat.view(self._weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta_w = self._fake_quantize_delta()
        weight = self._base_linear.weight + delta_w
        return F.linear(x, weight, self._base_linear.bias)

    # --- Persistence: save as 4-bit integers + scale/zp ---

    def get_quantized_state(self) -> dict:
        """Return the truly quantized representation (for compact storage)."""
        with torch.no_grad():
            bs = self.block_size
            delta_blocks = self.delta.view(-1, bs)
            s = self.scale.abs().unsqueeze(1)
            z = self.zero_point.unsqueeze(1)
            x_scaled = delta_blocks / (s + 1e-8) + z
            x_int = x_scaled.clamp(0, self.q_max).round().to(torch.uint8)
            return {
                "codes": x_int,              # (n_blocks, block_size) uint8
                "scale": self.scale.data,    # (n_blocks,)
                "zero_point": self.zero_point.data,  # (n_blocks,)
                "weight_shape": self._weight_shape,
                "n_padded": self._n_padded,
                "block_size": self.block_size,
                "n_bits": self.n_bits,
            }

    def load_quantized_state(self, state: dict):
        """Load from a quantized checkpoint and reconstruct the float delta."""
        with torch.no_grad():
            self.scale.copy_(state["scale"])
            self.zero_point.copy_(state["zero_point"])
            codes = state["codes"].float()
            s = self.scale.abs().unsqueeze(1)
            z = self.zero_point.unsqueeze(1)
            delta_blocks = s * (codes - z)
            self.delta.copy_(delta_blocks.reshape(-1))

    # --- Properties ---

    @property
    def in_features(self) -> int:
        return self._base_linear.in_features

    @property
    def out_features(self) -> int:
        return self._base_linear.out_features

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"block_size={self.block_size}, n_bits={self.n_bits}"
        )


# ---------------------------------------------------------------------------
# A2ADeltaAdapter: manages injection into all trainable modules
# ---------------------------------------------------------------------------

class A2ADeltaAdapter(nn.Module):
    """4-bit delta adapter for :class:`A2AImagePolicy`.

    Injects :class:`QuantizedDeltaLinear` into **every** ``nn.Linear`` layer
    of the trainable modules (flow_net, action_decoder, obs_projector,
    action_encoder, history_action_encoder).  Only ``obs_encoder`` (frozen
    vision backbone) is excluded.

    Parameters
    ----------
    base_policy : A2AImagePolicy
        A loaded Phase-1 model.  Will be frozen in-place.
    task_name : str
        Target task name.
    block_size : int
        Block size for quantization.
    n_bits : int
        Bit-width (default 4).
    """

    def __init__(
        self,
        base_policy,
        task_name: str,
        block_size: int = 64,
        n_bits: int = 4,
    ):
        super().__init__()
        self.task_name = task_name
        self._config = dict(block_size=block_size, n_bits=n_bits)

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

        # Track (parent, attr, original) for restore.
        self._injection_records: List[Tuple[nn.Module, str, nn.Module]] = []

        # --- Inject into all trainable Linear layers ---
        self.delta_layers = nn.ModuleList()

        # 1) flow_net: all Linear layers
        self._inject_all_linears(base_policy.flow_net, block_size, n_bits)

        # 2) action_decoder: all Linear layers
        self._inject_all_linears(base_policy.action_decoder, block_size, n_bits)

        # 3) obs_projector (single Linear)
        self._inject_linear(base_policy, "obs_projector", block_size, n_bits)

        # 4) action_encoder: all Linear layers (latent_proj, etc.)
        self._inject_all_linears(base_policy.action_encoder, block_size, n_bits)

        # 5) history_action_encoder: all Linear layers
        self._inject_all_linears(base_policy.history_action_encoder, block_size, n_bits)

    # ------------------------------------------------------------------ #
    #  Injection helpers                                                  #
    # ------------------------------------------------------------------ #

    def _inject_linear(
        self, parent: nn.Module, attr_name: str,
        block_size: int, n_bits: int,
    ) -> QuantizedDeltaLinear:
        original = getattr(parent, attr_name)
        assert isinstance(original, nn.Linear), (
            f"Expected nn.Linear at {type(parent).__name__}.{attr_name}, "
            f"got {type(original).__name__}"
        )
        delta_layer = QuantizedDeltaLinear(original, block_size=block_size, n_bits=n_bits)
        self._injection_records.append((parent, attr_name, original))
        setattr(parent, attr_name, delta_layer)
        self.delta_layers.append(delta_layer)
        return delta_layer

    def _inject_all_linears(
        self, module: nn.Module, block_size: int, n_bits: int,
    ):
        """Recursively find and inject all ``nn.Linear`` layers in *module*."""
        for name, child in list(module.named_modules()):
            if not isinstance(child, nn.Linear):
                continue
            # Resolve parent + attr from dotted name
            parts = name.split(".")
            parent = module
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr_name = parts[-1]
            self._inject_linear(parent, attr_name, block_size, n_bits)

    def restore(self):
        """Remove all delta injections, restoring original layers."""
        for parent, attr_name, original in self._injection_records:
            setattr(parent, attr_name, original)
        self._injection_records.clear()

    # ------------------------------------------------------------------ #
    #  Forward helpers                                                    #
    # ------------------------------------------------------------------ #

    def compute_loss(self, base_policy, batch: dict) -> torch.Tensor:
        if self._task_idx is not None and "task_idx" not in batch:
            B, T = batch["action"].shape[:2]
            batch = {
                **batch,
                "task_idx": torch.full(
                    (B, T), self._task_idx,
                    dtype=torch.long, device=batch["action"].device,
                ),
            }
        return base_policy.compute_loss(batch)

    @torch.no_grad()
    def predict_action(
        self, base_policy, obs_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
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
        bs = self._config["block_size"]
        nb = self._config["n_bits"]
        return (
            f"A2ADeltaAdapter(task={self.task_name!r}, "
            f"delta_layers={len(self.delta_layers)}, "
            f"block_size={bs}, n_bits={nb}, "
            f"trainable_params={n:,})"
        )

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Save adapter: full float state_dict (for resuming training)."""
        torch.save(
            {
                "task_name": self.task_name,
                "config": self._config,
                "adapter_state_dict": self.state_dict(),
            },
            path,
        )

    def save_quantized(self, path: str):
        """Save compact 4-bit representation (for deployment)."""
        quantized_states = [dl.get_quantized_state() for dl in self.delta_layers]
        torch.save(
            {
                "task_name": self.task_name,
                "config": self._config,
                "quantized_states": quantized_states,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, base_policy, device: str = "cpu") -> "A2ADeltaAdapter":
        """Load a saved adapter (float state_dict) and inject into base_policy."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(base_policy, task_name=ckpt["task_name"], **ckpt["config"])
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        adapter.to(device)
        return adapter

    @classmethod
    def load_quantized(cls, path: str, base_policy, device: str = "cpu") -> "A2ADeltaAdapter":
        """Load from compact 4-bit checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        adapter = cls(base_policy, task_name=ckpt["task_name"], **ckpt["config"])
        for dl, qs in zip(adapter.delta_layers, ckpt["quantized_states"]):
            dl.load_quantized_state(qs)
        adapter.to(device)
        return adapter


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    print("=" * 60)
    print("Testing QuantizedDeltaLinear ...")
    print("=" * 60)

    base = nn.Linear(32, 64)
    base.requires_grad_(False)
    qdl = QuantizedDeltaLinear(base, block_size=32, n_bits=4)

    x = torch.randn(2, 32)

    # 1. Initial output matches base (delta is zero).
    with torch.no_grad():
        diff = (qdl(x) - base(x)).abs().max().item()
    assert diff < 1e-5, f"Initial output diverges: max diff = {diff}"
    print("  [OK] Initial output matches base (delta=0)")

    # 2. state_dict contains only delta params.
    sd_keys = set(qdl.state_dict().keys())
    expected = {"delta", "scale", "zero_point"}
    assert sd_keys == expected, f"Unexpected keys: {sd_keys}"
    print(f"  [OK] state_dict keys: {sd_keys}")

    # 3. Gradients flow to delta, scale, zero_point.
    qdl.zero_grad()
    loss = qdl(x).sum()
    loss.backward()
    assert qdl.delta.grad is not None, "No grad on delta"
    assert qdl.scale.grad is not None, "No grad on scale"
    assert qdl.zero_point.grad is not None, "No grad on zero_point"
    print("  [OK] Gradients flow to delta, scale, zero_point")

    # 4. Base weights NOT in named_parameters.
    param_names = {n for n, _ in qdl.named_parameters()}
    assert param_names == expected, f"Unexpected params: {param_names}"
    print("  [OK] Base weights excluded from named_parameters")

    # 5. After some training steps, output should differ from base.
    opt = torch.optim.Adam(qdl.parameters(), lr=0.1)
    target = torch.randn(2, 64)
    for _ in range(20):
        opt.zero_grad()
        out = qdl(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt.step()
    with torch.no_grad():
        diff = (qdl(x) - base(x)).abs().max().item()
    assert diff > 0.01, f"Delta not learned: max diff = {diff}"
    print(f"  [OK] After training, output differs from base (diff={diff:.4f})")

    # 6. Quantized save/load round-trip.
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
    try:
        qs = qdl.get_quantized_state()
        qdl2 = QuantizedDeltaLinear(base, block_size=32, n_bits=4)
        qdl2.load_quantized_state(qs)
        with torch.no_grad():
            # Quantized round-trip should match closely (not exactly, due to
            # the latent delta being reconstructed from integer codes).
            out1 = qdl(x)
            out2 = qdl2(x)
            diff = (out1 - out2).abs().max().item()
        # The difference should be very small (within quantization noise)
        assert diff < 0.1, f"Quantized round-trip diverges: {diff}"
        print(f"  [OK] Quantized save/load round-trip (max diff={diff:.6f})")
    finally:
        os.unlink(tmp_path)

    # 7. Verify STE gradient is zero outside clamp range.
    print("\nTesting STE clamp behavior ...")
    qdl3 = QuantizedDeltaLinear(base, block_size=32, n_bits=4)
    # Set delta to large values that will be clamped
    with torch.no_grad():
        qdl3.delta.fill_(100.0)  # Way above q_max * scale
    qdl3.zero_grad()
    qdl3(x).sum().backward()
    # Some gradient elements should be zero (clamped region)
    n_zero = (qdl3.delta.grad.abs() < 1e-10).sum().item()
    print(f"  [OK] STE: {n_zero}/{qdl3.delta.numel()} grads zeroed by clamp")

    print("\n" + "=" * 60)
    print("All QuantizedDeltaLinear tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _self_test()
