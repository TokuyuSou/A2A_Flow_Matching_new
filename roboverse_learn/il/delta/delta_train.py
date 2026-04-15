#!/usr/bin/env python3
"""Train a 4-bit quantized delta adapter for an A2A policy on a single task.

Loads a Phase-1 multi-task checkpoint, freezes it, injects quantized delta
layers into every trainable Linear, and trains only the delta parameters.

Supports multi-GPU distributed training via ``torchrun``.

Usage
-----

Single GPU::

    python roboverse_learn/il/delta/delta_train.py \
        --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
        --task_name close_box \
        --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \
        --output_dir ./delta_adapters/close_box \
        --num_epochs 80

Multiple datasets (concatenated)::

    python roboverse_learn/il/delta/delta_train.py \
        --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
        --task_name close_box \
        --data_dir ./data_policy/close_box_v0_100.zarr ./data_policy/close_box_v1_50.zarr \
        --output_dir ./delta_adapters/close_box \
        --num_epochs 80

Multi-GPU (e.g. 4 GPUs)::

    torchrun --nproc_per_node=4 roboverse_learn/il/delta/delta_train.py \
        --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \
        --task_name close_box \
        --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \
        --output_dir ./delta_adapters/close_box \
        --num_epochs 80
"""

import argparse
import os
import pathlib
import sys

import dill
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from roboverse_learn.il.delta.delta_modules import A2ADeltaAdapter
from roboverse_learn.il.utils.visualization import plot_all_latent_visualizations


# ----------------------------------------------------------------------- #
#  Distributed helpers (same pattern as default_runner.py)                 #
# ----------------------------------------------------------------------- #


def _get_dist_info():
    """Return (rank, local_rank, world_size, is_distributed, is_main)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    is_main = rank == 0
    return rank, local_rank, world_size, is_distributed, is_main


class BatchSampler:
    """Yields numpy arrays of batch indices.

    Supports distributed training: when *world_size* > 1 each rank
    receives a disjoint partition of the data.  Call :meth:`set_epoch`
    before each epoch so that the shuffle is deterministic yet different
    across epochs.
    """

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        per_rank = data_size // world_size
        self.per_rank_size = per_rank
        self.num_batch = per_rank // batch_size

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic reshuffling across ranks."""
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            perm = rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)

        start = self.rank * self.per_rank_size
        rank_perm = perm[start : start + self.per_rank_size]

        usable = self.num_batch * self.batch_size
        rank_perm = rank_perm[:usable].reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield rank_perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int = 0,
    rank: int = 0,
    world_size: int = 1,
):
    """Create a DataLoader with distributed-aware BatchSampler."""
    sampler = BatchSampler(
        len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True,
        rank=rank, world_size=world_size,
    )

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers and num_workers > 0,
    )
    dataloader._dist_sampler = sampler
    return dataloader


# ----------------------------------------------------------------------- #
#  Helpers                                                                 #
# ----------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Train 4-bit delta adapter for A2A policy")

    # Required
    p.add_argument("--base_checkpoint", type=str, required=True,
                    help="Path to Phase-1 base checkpoint (.ckpt)")
    p.add_argument("--task_name", type=str, required=True,
                    help="Task name (must exist in base checkpoint's task_names)")
    p.add_argument("--data_dir", type=str, nargs="+", required=True,
                    help="Path(s) to single-task zarr dataset(s). "
                         "Multiple paths are concatenated for training.")
    p.add_argument("--output_dir", type=str, default=None,
                    help="Directory to save adapter checkpoints "
                         "(default: delta_adapters/{task_name}/{timestamp})")
    p.add_argument("--overwrite", action="store_true",
                    help="Allow writing into an existing output directory")

    # Delta quantization params
    p.add_argument("--block_size", type=int, default=32,
                    help="Block size for quantization (elements per block)")
    p.add_argument("--n_bits", type=int, default=4,
                    help="Bit-width for delta quantization")

    # Training hyper-parameters
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scale", type=float, default=0.3,
                    help="LR multiplier for scale/zero_point params")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--quant_warmup_epochs", type=int, default=5,
                    help="Epochs of float delta before enabling quantization")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=20,
                    help="Save a periodic checkpoint every N epochs (0=off)")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--max_val_steps", type=int, default=250)
    p.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader worker processes (increase to reduce CPU bottleneck)")

    # Logging
    p.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default="a2a_delta")
    p.add_argument("--wandb_name", type=str, default=None)

    return p.parse_args()


def load_base_policy(checkpoint_path: str, device: str):
    """Load the Phase-1 A2A policy (EMA or standard) from checkpoint.

    Avoids importing the full DefaultRunner (which pulls in metasim /
    evaluation dependencies) by directly instantiating the policy via
    Hydra and loading the state dict manually.
    """
    import copy
    import hydra

    payload = torch.load(
        open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu",
    )
    cfg = payload["cfg"]

    # Build model directly from policy config (no eval/sim deps needed)
    model = hydra.utils.instantiate(cfg.policy_config)
    model.load_state_dict(payload["state_dicts"]["model"])

    ema_model = None
    if cfg.train_config.training_params.use_ema and "ema_model" in payload["state_dicts"]:
        ema_model = copy.deepcopy(model)
        ema_model.load_state_dict(payload["state_dicts"]["ema_model"])

    policy = ema_model if ema_model is not None else model
    policy.to(device)
    policy.eval()
    return policy, cfg


def create_dataset(data_dirs, cfg, val_ratio: float = 0.02, batch_size: int = 32):
    """Create a single-task dataset matching Phase 1.

    Parameters
    ----------
    data_dirs : str or list[str]
        One or more zarr dataset paths.  When multiple paths are given the
        underlying replay buffers are merged into a single dataset via
        ``MultiTaskRobotImageDataset``.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    if len(data_dirs) == 1:
        from roboverse_learn.il.datasets.robot_image_dataset import RobotImageDataset
        dataset = RobotImageDataset(
            zarr_path=data_dirs[0],
            horizon=cfg.horizon,
            pad_before=cfg.n_obs_steps - 1,
            pad_after=cfg.n_action_steps - 1,
            seed=42,
            val_ratio=val_ratio,
            batch_size=batch_size,
        )
    else:
        from roboverse_learn.il.datasets.robot_image_dataset import MultiTaskRobotImageDataset
        dataset = MultiTaskRobotImageDataset(
            zarr_paths=data_dirs,
            horizon=cfg.horizon,
            pad_before=cfg.n_obs_steps - 1,
            pad_after=cfg.n_action_steps - 1,
            seed=42,
            val_ratio=val_ratio,
            batch_size=batch_size,
        )

    val_dataset = dataset.get_validation_dataset() if val_ratio > 0 else None
    return dataset, val_dataset


def build_optimizer(adapter, lr, lr_scale):
    """Separate parameter groups: delta params vs scale/zero_point."""
    delta_params = []
    quant_params = []

    for name, param in adapter.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".scale") or name.endswith(".zero_point"):
            quant_params.append(param)
        else:
            delta_params.append(param)

    param_groups = [
        {"params": delta_params, "lr": lr},
        {"params": quant_params, "lr": lr * lr_scale},
    ]
    return torch.optim.AdamW(param_groups, weight_decay=0.0)


# ----------------------------------------------------------------------- #
#  Quantization warmup: temporarily disable fake quantization
# ----------------------------------------------------------------------- #


def reinit_scales_from_delta(adapter):
    """Re-initialize scale/zero_point from learned delta statistics.

    Called once after warmup so that the quantization grid matches the
    actual delta distribution, avoiding the dead-gradient problem.
    """
    with torch.no_grad():
        for dl in adapter.delta_layers:
            bs = dl.block_size
            delta_blocks = dl.delta.view(-1, bs)  # (n_blocks, block_size)
            q_half = dl.q_max / 2

            # Per-block range
            block_max = delta_blocks.abs().max(dim=1).values.clamp(min=1e-8)

            # Set scale so that the full delta range fits within [-q_half, q_max-q_half] * scale
            new_scale = block_max / q_half
            dl.scale.copy_(new_scale)

            # Keep zero_point at integer center (no need to change)
    return new_scale


class _BypassQuantization:
    """Context manager that makes QuantizedDeltaLinear/Conv1d use float delta."""

    def __init__(self, adapter):
        self._adapter = adapter

    def __enter__(self):
        from roboverse_learn.il.delta.delta_modules import QuantizedDeltaLinear, QuantizedDeltaConv1d
        for dl in self._adapter.delta_layers:
            dl._orig_forward = dl.forward

            if isinstance(dl, QuantizedDeltaLinear):
                base = dl._base_linear

                def make_float_forward_linear(delta_layer, base_linear):
                    def float_forward(x):
                        import torch.nn.functional as _F
                        numel = delta_layer._weight_shape.numel()
                        delta_w = delta_layer.delta[:numel].view(delta_layer._weight_shape)
                        weight = base_linear.weight + delta_w
                        return _F.linear(x, weight, base_linear.bias)
                    return float_forward

                dl.forward = make_float_forward_linear(dl, base)

            elif isinstance(dl, QuantizedDeltaConv1d):
                base = dl._base_conv

                def make_float_forward_conv1d(delta_layer, base_conv):
                    def float_forward(x):
                        import torch.nn.functional as _F
                        numel = delta_layer._weight_shape.numel()
                        delta_w = delta_layer.delta[:numel].view(delta_layer._weight_shape)
                        weight = base_conv.weight + delta_w
                        return _F.conv1d(
                            x, weight, base_conv.bias,
                            stride=base_conv.stride,
                            padding=base_conv.padding,
                            dilation=base_conv.dilation,
                            groups=base_conv.groups,
                        )
                    return float_forward

                dl.forward = make_float_forward_conv1d(dl, base)
        return self

    def __exit__(self, *args):
        for dl in self._adapter.delta_layers:
            dl.forward = dl._orig_forward
            del dl._orig_forward


# ----------------------------------------------------------------------- #
#  Main                                                                    #
# ----------------------------------------------------------------------- #


def main():
    args = parse_args()

    # --- Distributed setup ---
    rank, local_rank, world_size, is_distributed, is_main = _get_dist_info()

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if is_distributed and is_main:
        print(f"Distributed training: world_size={world_size}")

    # --- Resolve output_dir ---
    if args.output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join("delta_adapters", args.task_name, timestamp)

    if is_main:
        if os.path.exists(args.output_dir) and not args.overwrite:
            if any(True for _ in os.scandir(args.output_dir)):
                print(
                    f"ERROR: Output directory already exists and is not empty:\n"
                    f"  {args.output_dir}\n"
                    f"Use --overwrite to force, or omit --output_dir for an "
                    f"auto-generated timestamped path."
                )
                sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)

    # Wait for rank 0 to create output directory
    if is_distributed:
        dist.barrier()

    # Use a consistent seed for model loading so all ranks start identical.
    torch.manual_seed(args.seed)

    # 1. Load base model
    if is_main:
        print(f"Loading base checkpoint: {args.base_checkpoint}")
    base_policy, cfg = load_base_policy(args.base_checkpoint, str(device))

    # 2. Create delta adapter
    adapter = A2ADeltaAdapter(
        base_policy,
        task_name=args.task_name,
        block_size=args.block_size,
        n_bits=args.n_bits,
    )
    adapter.to(device)
    if is_main:
        print(adapter.param_summary())

    # 3. Dataset and dataloader (distributed-aware)
    if is_main:
        for d in args.data_dir:
            print(f"Loading dataset: {d}")
    dataset, val_dataset = create_dataset(args.data_dir, cfg, args.val_ratio, args.batch_size)

    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
    )

    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
    if is_main:
        print(f"Dataset: {len(dataset)} train samples, {len(dataloader)} batches/epoch")
        if val_dataloader is not None:
            print(f"         {len(val_dataset)} val samples, {len(val_dataloader)} batches")

    # 4. Optimizer
    # Linear scaling rule: gradient averaging (ReduceOp.AVG) shrinks the
    # effective gradient by 1/world_size.  Scale LR to compensate so that
    # each optimizer step has the same effective magnitude as single-GPU.
    scaled_lr = args.lr * world_size
    optimizer = build_optimizer(adapter, scaled_lr, args.lr_scale)
    total_steps = args.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # 5. wandb (rank 0 only)
    import wandb
    wandb_run = None
    if is_main and args.wandb_mode != "disabled":
        run_name = args.wandb_name or f"delta_{args.task_name}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=args.wandb_mode,
            dir=args.output_dir,
            config={
                "base_checkpoint": args.base_checkpoint,
                "task_name": args.task_name,
                "data_dir": args.data_dir,
                "block_size": args.block_size,
                "n_bits": args.n_bits,
                "num_epochs": args.num_epochs,
                "lr": scaled_lr,
                "lr_base": args.lr,
                "lr_scale": args.lr_scale,
                "batch_size": args.batch_size,
                "grad_clip": args.grad_clip,
                "quant_warmup_epochs": args.quant_warmup_epochs,
                "seed": args.seed,
                "trainable_params": adapter.num_trainable_params(),
                "output_dir": args.output_dir,
                "world_size": world_size,
            },
        )

    # 6. Training loop
    # Per-rank seed so each rank samples different flow-matching timesteps,
    # increasing the diversity of the training signal per gradient-averaging step.
    torch.manual_seed(args.seed + rank)

    best_loss = float("inf")
    global_step = 0
    if is_main:
        print(f"\nStarting training: {args.num_epochs} epochs, "
              f"{total_steps} total steps\n")

    for epoch in range(1, args.num_epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        num_batches = 0

        # Reshuffle data partition each epoch for distributed training
        if hasattr(dataloader, '_dist_sampler'):
            dataloader._dist_sampler.set_epoch(epoch)

        # Warmup: disable quantization for first N epochs
        use_bypass = (args.quant_warmup_epochs > 0 and epoch <= args.quant_warmup_epochs)
        ctx = _BypassQuantization(adapter) if use_bypass else nullcontext()

        # Re-initialize scales right after warmup ends
        if args.quant_warmup_epochs > 0 and epoch == args.quant_warmup_epochs + 1:
            new_scale = reinit_scales_from_delta(adapter)
            if is_main:
                print(f"  Scale re-initialized from delta statistics "
                      f"(mean_scale={new_scale.mean().item():.2e})")

        with ctx:
            with tqdm.tqdm(
                dataloader,
                desc=f"Training epoch {epoch}",
                leave=False,
                mininterval=1.0,
                disable=not is_main,
            ) as tepoch:
                for raw_batch in tepoch:
                    batch = dataset.postprocess(raw_batch, str(device))
                    loss = adapter.compute_loss(base_policy, batch)

                    optimizer.zero_grad()
                    loss.backward()

                    # Synchronize gradients across ranks before stepping
                    if is_distributed:
                        for param in adapter.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            adapter.parameters(), args.grad_clip,
                        )
                    optimizer.step()
                    scheduler.step()

                    loss_val = loss.item()
                    epoch_loss += loss_val
                    num_batches += 1
                    tepoch.set_postfix(loss=f"{loss_val:.4f}")

                    if is_main and wandb_run is not None:
                        wandb_run.log(
                            {
                                "train_loss": loss_val,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                                "epoch": epoch,
                                "quant_active": not use_bypass,
                            },
                            step=global_step,
                        )
                    global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        # --- Validation & checkpointing (rank 0 only) ---
        if is_main:
            val_loss = None
            if val_dataloader is not None and epoch % args.val_every == 0:
                adapter.eval()
                val_ctx = _BypassQuantization(adapter) if use_bypass else nullcontext()
                with val_ctx:
                    val_losses = []
                    with torch.no_grad():
                        for val_idx, raw_val_batch in enumerate(val_dataloader):
                            val_batch = val_dataset.postprocess(raw_val_batch, str(device))
                            vloss = adapter.compute_loss(base_policy, val_batch)
                            val_losses.append(vloss.item())
                            if val_idx >= args.max_val_steps - 1:
                                break
                    if val_losses:
                        val_loss = sum(val_losses) / len(val_losses)

                    # --- Latent space visualization ---
                    if hasattr(base_policy, 'get_latents_for_visualization'):
                        try:
                            with torch.no_grad():
                                all_history_latents = []
                                all_future_latents = []
                                max_samples = 500
                                first_batch = None

                                for raw_viz_batch in val_dataloader:
                                    viz_batch = val_dataset.postprocess(raw_viz_batch, str(device))
                                    if (
                                        getattr(adapter, "_task_idx", None) is not None
                                        and "task_idx" not in viz_batch
                                    ):
                                        B, T = viz_batch["action"].shape[:2]
                                        viz_batch = {
                                            **viz_batch,
                                            "task_idx": torch.full(
                                                (B, T), adapter._task_idx,
                                                dtype=torch.long,
                                                device=viz_batch["action"].device,
                                            ),
                                        }
                                    if first_batch is None:
                                        first_batch = viz_batch
                                    h_lat, f_lat = base_policy.get_latents_for_visualization(viz_batch)
                                    all_history_latents.append(h_lat.cpu())
                                    all_future_latents.append(f_lat.cpu())
                                    if sum(h.shape[0] for h in all_history_latents) >= max_samples:
                                        break

                                history_latents = torch.cat(all_history_latents, dim=0)[:max_samples]
                                future_latents = torch.cat(all_future_latents, dim=0)[:max_samples]

                                trajectories = None
                                trajectory_targets = None
                                if hasattr(base_policy, 'get_flow_trajectories') and first_batch is not None:
                                    trajectories, trajectory_targets = base_policy.get_flow_trajectories(
                                        first_batch, n_samples=5,
                                    )

                                viz_dir = pathlib.Path(args.output_dir) / "latent_viz"
                                viz_results = plot_all_latent_visualizations(
                                    history_latents=history_latents,
                                    future_latents=future_latents,
                                    epoch=epoch,
                                    save_dir=str(viz_dir),
                                    trajectories=trajectories,
                                    trajectory_targets=trajectory_targets,
                                )
                                print(f"  Latent viz saved to {viz_dir}  "
                                      f"avg_tsne_dist={viz_results['avg_tsne_distance']:.2f}")

                                if wandb_run is not None:
                                    latent_metrics = {
                                        "latent/avg_tsne_distance": viz_results['avg_tsne_distance'],
                                    }
                                    if 'flow_end_to_target_dist' in viz_results:
                                        latent_metrics["latent/flow_end_to_target_dist"] = viz_results['flow_end_to_target_dist']
                                    wandb_run.log(latent_metrics, step=global_step)
                        except Exception as e:
                            print(f"  Warning: latent visualization failed: {e}")

            # --- Print ---
            quant_tag = " [float warmup]" if use_bypass else ""
            line = (
                f"Epoch {epoch:3d}/{args.num_epochs}  "
                f"train_loss={avg_loss:.6f}"
            )
            if val_loss is not None:
                line += f"  val_loss={val_loss:.6f}"
            line += f"  lr={lr_now:.2e}{quant_tag}"
            print(line)

            # --- Per-epoch logging ---
            epoch_log = {
                "epoch_train_loss": avg_loss,
                "best_loss": min(best_loss, avg_loss),
                "epoch": epoch,
            }
            if val_loss is not None:
                epoch_log["val_loss"] = val_loss
            if wandb_run is not None:
                wandb_run.log(epoch_log, step=global_step)

            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                adapter.save(os.path.join(args.output_dir, "best.pt"))

            # Save periodic
            if args.save_every > 0 and epoch % args.save_every == 0:
                adapter.save(os.path.join(args.output_dir, f"epoch_{epoch}.pt"))

        # Synchronize all ranks before next epoch
        if is_distributed:
            dist.barrier()

    # Save last (both float and quantized) — rank 0 only
    if is_main:
        adapter.save(os.path.join(args.output_dir, "last.pt"))
        adapter.save_quantized(os.path.join(args.output_dir, "last_quantized.pt"))

        print(f"\nDone. Best training loss: {best_loss:.6f}")
        print(f"Adapters saved to: {args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()


# Python 3.7+ contextlib.nullcontext
try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


if __name__ == "__main__":
    # Initialize distributed training when launched via torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    try:
        main()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
