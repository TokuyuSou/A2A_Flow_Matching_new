#!/usr/bin/env python3
"""Train a LoRA adapter for an A2A policy on a single task.

Loads a Phase-1 multi-task checkpoint, freezes it, injects LoRA layers, and
trains only the LoRA parameters on a single-task dataset.

Usage
-----
::

    python roboverse_learn/il/lora/lora_train.py \\
        --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \\
        --task_name close_box \\
        --data_dir ./data_policy/close_box_rlbench_v0_100.zarr \\
        --output_dir ./lora_adapters/close_box \\
        --num_epochs 80
"""

import argparse
import os
import sys

import dill
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Ensure the repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from roboverse_learn.il.lora.lora_modules import A2ALoRAAdapter


# ----------------------------------------------------------------------- #
#  Helpers                                                                 #
# ----------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA adapter for A2A policy")

    # Required
    p.add_argument(
        "--base_checkpoint", type=str, required=True,
        help="Path to Phase-1 base checkpoint (.ckpt)",
    )
    p.add_argument(
        "--task_name", type=str, required=True,
        help="Task name (must exist in base checkpoint's task_names)",
    )
    p.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to single-task zarr dataset",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save adapter checkpoints",
    )

    # LoRA ranks
    p.add_argument("--flow_mlp_rank", type=int, default=8)
    p.add_argument("--decoder_rank", type=int, default=4)
    p.add_argument("--task_rank", type=int, default=4)
    p.add_argument("--out_proj_rank", type=int, default=4)

    # Training hyper-parameters
    p.add_argument("--num_epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=20,
                    help="Save a periodic checkpoint every N epochs (0 = off)")
    p.add_argument("--val_ratio", type=float, default=0.02,
                    help="Fraction of episodes held out for validation")
    p.add_argument("--val_every", type=int, default=5,
                    help="Run validation every N epochs")
    p.add_argument("--max_val_steps", type=int, default=250,
                    help="Max batches per validation run")

    # Logging
    p.add_argument("--wandb_mode", type=str, default="online",
                    choices=["online", "offline", "disabled"],
                    help="wandb logging mode")
    p.add_argument("--wandb_project", type=str, default="a2a_lora",
                    help="wandb project name")
    p.add_argument("--wandb_name", type=str, default=None,
                    help="wandb run name (default: lora_{task_name})")

    return p.parse_args()


def load_base_policy(checkpoint_path: str, device: str):
    """Load the Phase-1 A2A policy (EMA or standard) from *checkpoint_path*.

    Heavy dependencies (DefaultRunner, metasim, hydra, …) are imported lazily
    so that ``--help`` and lightweight checks do not trigger them.
    """
    from roboverse_learn.il.runners.default_runner import DefaultRunner

    payload = torch.load(
        open(checkpoint_path, "rb"), pickle_module=dill, map_location=device,
    )
    cfg = payload["cfg"]

    runner = DefaultRunner(cfg)
    runner.load_payload(payload, exclude_keys=None, include_keys=None)

    if cfg.train_config.training_params.use_ema:
        policy = runner.ema_model
    else:
        policy = runner.model

    policy.to(device)
    policy.eval()
    return policy, cfg


def create_dataset(data_dir: str, cfg, val_ratio: float = 0.02):
    """Create a single-task :class:`RobotImageDataset` matching Phase 1.

    Imported lazily to avoid pulling in the full runner dependency chain.
    Returns ``(train_dataset, val_dataset)``.  ``val_dataset`` is ``None``
    when *val_ratio* is 0.
    """
    from roboverse_learn.il.datasets.robot_image_dataset import RobotImageDataset

    dataset = RobotImageDataset(
        zarr_path=data_dir,
        horizon=cfg.horizon,
        pad_before=cfg.n_obs_steps - 1,
        pad_after=cfg.n_action_steps - 1,
        seed=42,
        val_ratio=val_ratio,
    )
    val_dataset = dataset.get_validation_dataset() if val_ratio > 0 else None
    return dataset, val_dataset


# ----------------------------------------------------------------------- #
#  Main                                                                    #
# ----------------------------------------------------------------------- #


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. Load base model ------------------------------------------------
    print(f"Loading base checkpoint: {args.base_checkpoint}")
    base_policy, cfg = load_base_policy(args.base_checkpoint, args.device)

    # 2. Create LoRA adapter (freezes base, injects LoRA) ---------------
    adapter = A2ALoRAAdapter(
        base_policy,
        task_name=args.task_name,
        flow_mlp_rank=args.flow_mlp_rank,
        decoder_rank=args.decoder_rank,
        task_rank=args.task_rank,
        out_proj_rank=args.out_proj_rank,
    )
    adapter.to(args.device)
    print(adapter.param_summary())

    # 3. Dataset and dataloader -----------------------------------------
    print(f"Loading dataset: {args.data_dir}")
    dataset, val_dataset = create_dataset(args.data_dir, cfg, args.val_ratio)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    print(f"Dataset: {len(dataset)} train samples, {len(dataloader)} batches/epoch")
    if val_dataloader is not None:
        print(f"         {len(val_dataset)} val samples, {len(val_dataloader)} batches")

    # 4. Optimiser (adapter params only) --------------------------------
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=0.0,
    )
    total_steps = args.num_epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    # 5. wandb ----------------------------------------------------------
    import wandb

    wandb_run = None
    if args.wandb_mode != "disabled":
        run_name = args.wandb_name or f"lora_{args.task_name}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            mode=args.wandb_mode,
            dir=args.output_dir,
            config={
                "base_checkpoint": args.base_checkpoint,
                "task_name": args.task_name,
                "data_dir": args.data_dir,
                "flow_mlp_rank": args.flow_mlp_rank,
                "decoder_rank": args.decoder_rank,
                "task_rank": args.task_rank,
                "out_proj_rank": args.out_proj_rank,
                "num_epochs": args.num_epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "grad_clip": args.grad_clip,
                "seed": args.seed,
                "trainable_params": adapter.num_trainable_params(),
                "output_dir": args.output_dir,
            },
        )

    # 6. Training loop --------------------------------------------------
    best_loss = float("inf")
    global_step = 0
    print(f"\nStarting training: {args.num_epochs} epochs, "
          f"{total_steps} total steps\n")

    for epoch in range(1, args.num_epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        num_batches = 0

        for raw_batch in dataloader:
            batch = dataset.postprocess(raw_batch, args.device)

            loss = adapter.compute_loss(base_policy, batch)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), args.grad_clip,
                )
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            epoch_loss += loss_val
            num_batches += 1

            # Per-step logging
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train_loss": loss_val,
                        "lr": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        lr_now = scheduler.get_last_lr()[0]

        # --- Validation ---
        val_loss = None
        if (
            val_dataloader is not None
            and epoch % args.val_every == 0
        ):
            adapter.eval()
            val_losses = []
            with torch.no_grad():
                for val_idx, raw_val_batch in enumerate(val_dataloader):
                    val_batch = val_dataset.postprocess(
                        raw_val_batch, args.device,
                    )
                    vloss = adapter.compute_loss(base_policy, val_batch)
                    val_losses.append(vloss.item())
                    if val_idx >= args.max_val_steps - 1:
                        break
            if val_losses:
                val_loss = sum(val_losses) / len(val_losses)

        # --- Print ---
        line = (
            f"Epoch {epoch:3d}/{args.num_epochs}  "
            f"train_loss={avg_loss:.6f}"
        )
        if val_loss is not None:
            line += f"  val_loss={val_loss:.6f}"
        line += f"  lr={lr_now:.2e}"
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
            adapter.save(
                os.path.join(args.output_dir, f"epoch_{epoch}.pt"),
            )

    # Save last
    adapter.save(os.path.join(args.output_dir, "last.pt"))

    print(f"\nDone. Best training loss: {best_loss:.6f}")
    print(f"Adapters saved to: {args.output_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
