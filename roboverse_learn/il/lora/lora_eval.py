#!/usr/bin/env python3
"""Evaluate a LoRA-adapted A2A policy on RLBench tasks.

Loads a Phase-1 base checkpoint and a LoRA adapter, injects the adapter,
then runs evaluation episodes in RLBench (CoppeliaSim).

Usage
-----
::

    python roboverse_learn/il/lora/lora_eval.py \\
        --base_checkpoint ./il_outputs/a2a/multi_task/checkpoints/200.ckpt \\
        --adapter_path ./lora_adapters/close_box/best.pt \\
        --task_name close_box \\
        --num_episodes 25 \\
        --headless
"""

import argparse
import multiprocessing as mp
import os
import sys
from collections import deque

import imageio
import numpy as np
import torch

# Ensure repo root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from roboverse_learn.il.lora.lora_modules import A2ALoRAAdapter

# Camera attribute mapping (copied from rlbench_eval to avoid heavy import).
CAMERA_RGB_ATTRS = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
    "overhead": "overhead_rgb",
    "wrist": "wrist_rgb",
}


# ----------------------------------------------------------------------- #
#  Worker                                                                  #
# ----------------------------------------------------------------------- #


def run_episodes(worker_id, episode_ids, task_name, eval_dir, args, result_queue):
    """Worker: load base + adapter, launch CoppeliaSim, run assigned episodes."""
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointPosition
    from rlbench.action_modes.gripper_action_modes import GripperJointPosition
    from rlbench.environment import Environment
    from rlbench import utils as rlbench_utils

    # Lazy import of helpers that depend on DefaultRunner / metasim.
    from roboverse_learn.il.rlbench_eval import (
        build_obs_config, image_to_frame, load_policy,
        obs_to_image, obs_to_state, stack_obs,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Load base policy + LoRA adapter ---
    base_policy, cfg, n_obs_steps, n_action_steps = load_policy(
        args.base_checkpoint, str(device),
    )
    adapter = A2ALoRAAdapter.load(
        args.adapter_path, base_policy, device=str(device),
    )
    adapter.eval()

    # Set task embedding for evaluation.
    if adapter._task_idx is not None:
        base_policy.set_eval_task(task_name)

    # --- RLBench environment ---
    obs_config = build_obs_config(args.camera, args.image_size)
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(absolute_mode=True),
        gripper_action_mode=GripperJointPosition(absolute_mode=True),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=args.headless,
        dataset_root=args.dataset_root if args.dataset_root else "",
    )
    env.launch()

    task_class = rlbench_utils.name_to_task_class(task_name)
    task_env = env.get_task(task_class)
    task_env.set_variation(args.variation)

    video_dir = os.path.join(eval_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    results = []

    for ep in episode_ids:
        descriptions, obs = task_env.reset()
        obs_history: deque = deque(maxlen=n_obs_steps + 1)
        action_cache: list = []
        frames: list = []

        state = obs_to_state(obs, args.camera)
        image = obs_to_image(obs, args.camera, args.image_size)
        frames.append(image_to_frame(image))
        obs_history.append({"agent_pos": state, "head_cam": image})

        success = False

        for step in range(args.max_steps):
            if len(action_cache) == 0:
                stacked = stack_obs(obs_history, n_obs_steps)
                obs_input = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device)
                    for k, v in stacked.items()
                }
                with torch.no_grad():
                    result = base_policy.predict_action(obs_input)
                    action_chunk = result["action"].detach().cpu().numpy()[0]
                action_cache = list(action_chunk)

            action_9d = action_cache.pop(0)
            arm_action = action_9d[:7]
            gripper_action = np.array([np.mean(action_9d[7:9])])
            rlbench_action = np.concatenate([arm_action, gripper_action])

            try:
                obs, reward, done = task_env.step(rlbench_action)
            except Exception as e:
                print(f"  [Worker {worker_id}] Episode {ep}: step {step} error: {e}")
                break

            state = obs_to_state(obs, args.camera)
            image = obs_to_image(obs, args.camera, args.image_size)
            frames.append(image_to_frame(image))
            obs_history.append({"agent_pos": state, "head_cam": image})

            if reward > 0:
                success = True
                break

        num_steps = step + 1
        tag = "success" if success else "fail"
        imageio.mimwrite(
            os.path.join(video_dir, f"ep{ep:03d}_{tag}.mp4"), frames, fps=30,
        )
        results.append((ep, success, num_steps))
        print(
            f"  [Worker {worker_id}] Episode {ep}: "
            f"{'SUCCESS' if success else 'FAIL'} (steps={num_steps})"
        )

    env.shutdown()
    result_queue.put(results)


# ----------------------------------------------------------------------- #
#  Orchestration                                                           #
# ----------------------------------------------------------------------- #


def evaluate(task_name: str, eval_dir: str, args):
    """Run evaluation episodes (optionally multi-worker) for one task."""
    os.makedirs(os.path.join(eval_dir, "videos"), exist_ok=True)

    num_workers = max(1, args.num_workers)
    all_episodes = list(range(args.num_episodes))
    chunks = [all_episodes[i::num_workers] for i in range(num_workers)]

    if num_workers == 1:
        result_queue = mp.Queue()
        run_episodes(0, chunks[0], task_name, eval_dir, args, result_queue)
        all_results = result_queue.get()
    else:
        print(
            f"Launching {num_workers} parallel workers "
            f"for {args.num_episodes} episodes ..."
        )
        result_queue = mp.Queue()
        processes = []
        for wid, chunk in enumerate(chunks):
            if not chunk:
                continue
            p = mp.Process(
                target=run_episodes,
                args=(wid, chunk, task_name, eval_dir, args, result_queue),
            )
            p.start()
            processes.append(p)

        all_results = []
        for _ in processes:
            all_results.extend(result_queue.get())
        for p in processes:
            p.join()

    all_results.sort(key=lambda x: x[0])
    return all_results


# ----------------------------------------------------------------------- #
#  CLI                                                                     #
# ----------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA-adapted A2A policy on RLBench",
    )
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--camera", type=str, default="front",
                        choices=list(CAMERA_RGB_ATTRS.keys()))
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    if args.eval_dir is None:
        args.eval_dir = os.path.join(
            os.path.dirname(args.adapter_path), "eval",
        )

    # --- Validate ---
    print(f"Base checkpoint : {args.base_checkpoint}")
    print(f"Adapter         : {args.adapter_path}")
    print(f"Task            : {args.task_name}")

    # --- Evaluate ---
    results = evaluate(args.task_name, args.eval_dir, args)

    successes = sum(1 for _, s, _ in results if s)
    total = len(results)

    print(f"\n{'=' * 50}")
    print(f"Task: {args.task_name} (variation {args.variation})")
    print(f"Adapter: {args.adapter_path}")
    print(f"Success rate: {successes}/{total} = {successes / total:.1%}")
    print(f"{'=' * 50}")

    # Save results
    results_path = os.path.join(args.eval_dir, "results.txt")
    with open(results_path, "w") as f:
        for ep, success, steps in results:
            f.write(
                f"Episode {ep:03d}: "
                f"{'SUCCESS' if success else 'FAIL'} (steps={steps})\n"
            )
        f.write(f"\nTask: {args.task_name} (variation {args.variation})\n")
        f.write(f"Base: {args.base_checkpoint}\n")
        f.write(f"Adapter: {args.adapter_path}\n")
        f.write(f"Success rate: {successes}/{total} = {successes / total:.1%}\n")
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
