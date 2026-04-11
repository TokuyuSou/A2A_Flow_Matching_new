"""Evaluate a trained A2A policy on RLBench tasks.

This script drives RLBench's CoppeliaSim environment directly, converting
observations to the format the A2A policy expects and converting policy
outputs back to RLBench actions.

Supports multi-process evaluation via --num_workers to run episodes in
parallel, each with its own CoppeliaSim instance.

Usage
-----
# Single task:
python roboverse_learn/il/rlbench_eval.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --task_name reach_target \
    --num_episodes 25 \
    --num_workers 4 \
    --headless

# Multiple tasks (multi-task checkpoint):
python roboverse_learn/il/rlbench_eval.py \
    --checkpoint /path/to/checkpoint.ckpt \
    --task_name "reach_target+close_jar" \
    --num_episodes 25 \
    --headless
"""

import argparse
import multiprocessing as mp
import os
import sys
from collections import deque

import dill
import imageio
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# RLBench imports (require CoppeliaSim / PyRep installed)
# ---------------------------------------------------------------------------
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import GripperJointPosition
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench import utils as rlbench_utils

# ---------------------------------------------------------------------------
# A2A imports
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from roboverse_learn.il.runners.default_runner import DefaultRunner  # noqa: E402


CAMERA_RGB_ATTRS = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
    "overhead": "overhead_rgb",
    "wrist": "wrist_rgb",
}


def build_obs_config(camera: str, image_size: int) -> ObservationConfig:
    """Create an ObservationConfig that only enables the chosen camera + low-dim."""
    off_cam = CameraConfig(rgb=False, depth=False, point_cloud=False, mask=False)
    on_cam = CameraConfig(
        rgb=True, depth=False, point_cloud=False, mask=False,
        image_size=(image_size, image_size),
    )

    cam_map = {
        "front": "front_camera",
        "left_shoulder": "left_shoulder_camera",
        "right_shoulder": "right_shoulder_camera",
        "overhead": "overhead_camera",
        "wrist": "wrist_camera",
    }

    kwargs = {v: off_cam for v in cam_map.values()}
    kwargs[cam_map[camera]] = on_cam

    return ObservationConfig(
        **kwargs,
        joint_velocities=False,
        joint_positions=True,
        joint_forces=False,
        gripper_open=True,
        gripper_pose=True,
        gripper_matrix=False,
        gripper_joint_positions=True,
        gripper_touch_forces=False,
        task_low_dim_state=False,
    )


def obs_to_state(obs, camera: str) -> np.ndarray:
    """Convert RLBench Observation to 9D state vector."""
    jp = np.array(obs.joint_positions, dtype=np.float32)
    if obs.gripper_joint_positions is not None and len(obs.gripper_joint_positions) >= 2:
        gripper = np.array(obs.gripper_joint_positions[:2], dtype=np.float32)
    else:
        g = float(obs.gripper_open) if obs.gripper_open is not None else 0.0
        gripper = np.array([g * 0.04, g * 0.04], dtype=np.float32)
    return np.concatenate([jp, gripper])


def obs_to_image(obs, camera: str, image_size: int) -> np.ndarray:
    """Extract RGB image from Observation and return (3, H, W) float tensor."""
    rgb = getattr(obs, CAMERA_RGB_ATTRS[camera])
    if isinstance(rgb, str):
        rgb = np.array(Image.open(rgb).resize((image_size, image_size)))
    if rgb.shape[0] != image_size or rgb.shape[1] != image_size:
        rgb = np.array(Image.fromarray(rgb).resize((image_size, image_size)))
    return np.moveaxis(rgb[..., :3], -1, 0).astype(np.float32) / 255.0


def image_to_frame(image: np.ndarray) -> np.ndarray:
    """Convert CHW float [0,1] image to HWC uint8 frame for video."""
    return (image.transpose(1, 2, 0) * 255).astype(np.uint8)


def load_policy(checkpoint_path: str, device: str):
    """Load A2A policy from checkpoint."""
    import hydra
    from omegaconf import OmegaConf

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location=device)
    cfg = payload["cfg"]

    runner = DefaultRunner(cfg)
    runner.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = runner.model
    if cfg.train_config.training_params.use_ema:
        policy = runner.ema_model

    policy.to(device)
    policy.eval()

    n_obs_steps = getattr(policy, "n_obs_steps", cfg.n_obs_steps)
    n_action_steps = getattr(policy, "n_action_steps", cfg.n_action_steps)

    return policy, cfg, n_obs_steps, n_action_steps


def stack_obs(obs_history, n_steps: int):
    """Stack last n_steps observations, padding with first if needed."""
    result = {}
    obs_list = list(obs_history)
    for key in obs_list[-1]:
        vals = [o[key] for o in obs_list]
        arr = np.zeros((n_steps,) + vals[-1].shape, dtype=vals[-1].dtype)
        start = -min(n_steps, len(vals))
        arr[start:] = np.array(vals[start:])
        if n_steps > len(vals):
            arr[:start] = arr[start]
        result[key] = arr
    return result


def run_episodes(worker_id, episode_ids, task_name, eval_dir, args, result_queue):
    """Worker function: launch its own CoppeliaSim, run assigned episodes."""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy, cfg, n_obs_steps, n_action_steps = load_policy(args.checkpoint, str(device))

    # Set task embedding for evaluation (multi-task models)
    if hasattr(policy, 'set_eval_task'):
        policy.set_eval_task(task_name)

    obs_config = build_obs_config(args.camera, args.image_size)
    action_mode = MoveArmThenGripper(
        arm_action_mode=JointPosition(absolute_mode=True),
        gripper_action_mode=GripperJointPosition(absolute_mode=True),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=args.headless,
        dataset_root=args.dataset_root if args.dataset_root else '',
    )
    env.launch()

    task_class = rlbench_utils.name_to_task_class(task_name)
    task_env = env.get_task(task_class)
    task_env.set_variation(args.variation)

    video_dir = os.path.join(eval_dir, "videos")
    results = []

    for ep in episode_ids:
        descriptions, obs = task_env.reset()
        obs_history = deque(maxlen=n_obs_steps + 1)
        action_cache = []
        frames = []

        state = obs_to_state(obs, args.camera)
        image = obs_to_image(obs, args.camera, args.image_size)
        frames.append(image_to_frame(image))

        obs_dict = {"agent_pos": state, "head_cam": image}
        obs_history.append(obs_dict)

        success = False

        for step in range(args.max_steps):
            if len(action_cache) == 0:
                stacked = stack_obs(obs_history, n_obs_steps)
                obs_input = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device)
                    for k, v in stacked.items()
                }
                with torch.no_grad():
                    action_pred = policy.predict_action(obs_input)
                    action_chunk = action_pred["action"].detach().cpu().numpy()[0]
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
            obs_dict = {"agent_pos": state, "head_cam": image}
            obs_history.append(obs_dict)

            if reward > 0:
                success = True
                break

        num_steps = step + 1
        tag = "success" if success else "fail"
        imageio.mimwrite(os.path.join(video_dir, f"ep{ep:03d}_{tag}.mp4"), frames, fps=30)

        results.append((ep, success, num_steps))
        print(f"  [Worker {worker_id}] Episode {ep}: {'SUCCESS' if success else 'FAIL'} "
              f"(steps={num_steps})")

    env.shutdown()
    result_queue.put(results)


def evaluate_single_task(task_name, eval_dir, args):
    """Run evaluation for a single task. Returns list of (ep, success, steps)."""
    os.makedirs(os.path.join(eval_dir, "videos"), exist_ok=True)

    num_workers = max(1, args.num_workers)
    all_episodes = list(range(args.num_episodes))
    chunks = [all_episodes[i::num_workers] for i in range(num_workers)]

    if num_workers == 1:
        result_queue = mp.Queue()
        run_episodes(0, chunks[0], task_name, eval_dir, args, result_queue)
        all_results = result_queue.get()
    else:
        print(f"Launching {num_workers} parallel workers for {args.num_episodes} episodes...")
        result_queue = mp.Queue()
        processes = []
        for wid, chunk in enumerate(chunks):
            if len(chunk) == 0:
                continue
            p = mp.Process(target=run_episodes,
                           args=(wid, chunk, task_name, eval_dir, args, result_queue))
            p.start()
            processes.append(p)

        all_results = []
        for _ in processes:
            all_results.extend(result_queue.get())
        for p in processes:
            p.join()

    all_results.sort(key=lambda x: x[0])
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate A2A policy on RLBench.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True,
                        help="RLBench task name(s). Use '+' to separate multiple tasks "
                             "(e.g. 'reach_target+close_jar')")
    parser.add_argument("--num_episodes", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Max steps per episode")
    parser.add_argument("--camera", type=str, default="front",
                        choices=list(CAMERA_RGB_ATTRS.keys()))
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_root", type=str, default="",
                        help="RLBench dataset root (for reset_to_demo support)")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Output directory for videos and results. "
                             "Defaults to <checkpoint_dir>/../eval.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel CoppeliaSim workers.")
    args = parser.parse_args()

    task_list = args.task_name.split('+')
    multi_task = len(task_list) > 1

    # Default eval_dir: <task_dir>/eval
    if args.eval_dir is None:
        task_dir = os.path.dirname(os.path.dirname(args.checkpoint))
        args.eval_dir = os.path.join(task_dir, "eval")

    # --- Validate task names against checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, _, _, _ = load_policy(args.checkpoint, "cpu")
    available_tasks = getattr(policy, 'task_names', None)

    if available_tasks is not None:
        # Multi-task model: validate requested tasks
        for t in task_list:
            if t not in available_tasks:
                print(f"\nERROR: Task '{t}' not found in checkpoint.\n"
                      f"Available tasks: {available_tasks}\n"
                      f"Use '+' to join multiple tasks, e.g. --task_name '{'+'.join(available_tasks)}'")
                sys.exit(1)
        print(f"Multi-task checkpoint. Available tasks: {available_tasks}")
        if not multi_task and len(available_tasks) > 1:
            print(f"TIP: This checkpoint supports {len(available_tasks)} tasks. "
                  f"To evaluate all, use: --task_name '{'+'.join(available_tasks)}'")
    else:
        if multi_task:
            print("WARNING: This is a single-task checkpoint (no task embeddings). "
                  "Each task will be evaluated without task conditioning.")

    del policy

    # --- Evaluate each task ---
    summary = {}

    for task_name in task_list:
        print(f"\n{'=' * 50}")
        print(f"Evaluating task: {task_name}")
        print(f"{'=' * 50}")

        task_eval_dir = os.path.join(args.eval_dir, task_name) if multi_task else args.eval_dir
        results = evaluate_single_task(task_name, task_eval_dir, args)

        successes = sum(1 for _, s, _ in results if s)
        total = len(results)
        summary[task_name] = (successes, total)

        print(f"\n{'=' * 50}")
        print(f"Task: {task_name} (variation {args.variation})")
        print(f"Success rate: {successes}/{total} = {successes / total:.1%}")
        if args.num_workers > 1:
            print(f"Workers: {args.num_workers}")
        print(f"{'=' * 50}")

        # Save per-task results
        results_path = os.path.join(task_eval_dir, "results.txt")
        with open(results_path, "w") as f:
            for ep, success, steps in results:
                f.write(f"Episode {ep:03d}: {'SUCCESS' if success else 'FAIL'} (steps={steps})\n")
            f.write(f"\nTask: {task_name} (variation {args.variation})\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Success rate: {successes}/{total} = {successes / total:.1%}\n")
        print(f"Results saved to {results_path}")

    # Overall summary for multi-task
    if multi_task:
        print(f"\n{'=' * 50}")
        print("Overall Summary")
        print(f"{'=' * 50}")
        total_success = 0
        total_episodes = 0
        for t, (s, n) in summary.items():
            print(f"  {t}: {s}/{n} = {s / n:.1%}")
            total_success += s
            total_episodes += n
        print(f"  Average: {total_success}/{total_episodes} = {total_success / total_episodes:.1%}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
