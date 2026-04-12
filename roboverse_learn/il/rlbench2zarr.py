"""Convert RLBench demonstration datasets to the Zarr format expected by A2A.

Usage examples
--------------
# Single task, front camera, 256x256, 50 demos from variation 0:
python roboverse_learn/il/rlbench2zarr.py \
    --dataset_root /path/to/rlbench_data \
    --task_name reach_target \
    --num_demos 50

# Use a different camera and image size:
python roboverse_learn/il/rlbench2zarr.py \
    --dataset_root /path/to/rlbench_data \
    --task_name close_jar \
    --camera front \
    --image_size 128 \
    --variation 0

Zarr output layout
------------------
  data/
    head_camera : (T_total, 3, H, W) uint8   – RGB from the chosen camera
    state       : (T_total, 9)       float32  – joint_positions(7) + gripper(2)
    action      : (T_total, 9)       float32  – target joint positions
  meta/
    episode_ends: (N_episodes,)      int64    – cumulative frame counts
"""

import argparse
import os
import pickle
import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm

# Camera attribute name mapping (RLBench Observation attribute names)
CAMERA_RGB_ATTRS = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
    "overhead": "overhead_rgb",
    "wrist": "wrist_rgb",
}

# Folder names for each camera (from rlbench.backend.const)
CAMERA_RGB_FOLDERS = {
    "front": "front_rgb",
    "left_shoulder": "left_shoulder_rgb",
    "right_shoulder": "right_shoulder_rgb",
    "overhead": "overhead_rgb",
    "wrist": "wrist_rgb",
}

IMAGE_FORMAT = "%d.png"


class _PermissiveUnpickler(pickle.Unpickler):
    """Unpickler that handles missing classes gracefully.

    RLBench pickles reference ``rlbench.backend.observation.Observation``,
    which transitively requires PyRep / CoppeliaSim.  If those are not
    installed we still want to load the data, so we fall back to a simple
    namespace object when the real class is unavailable.
    """

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Return a generic class that accepts any attributes
            return type(name, (), {"__init__": lambda self, **kw: self.__dict__.update(kw)})


def load_demo_low_dim(episode_path: str):
    """Load low-dimensional observations from a single episode pickle."""
    pkl_path = os.path.join(episode_path, "low_dim_obs.pkl")
    with open(pkl_path, "rb") as f:
        obs_list = _PermissiveUnpickler(f).load()
    return obs_list


def load_demo_images(episode_path: str, camera: str, image_size: int, num_steps: int):
    """Load and resize RGB images for one episode from the chosen camera."""
    folder = os.path.join(episode_path, CAMERA_RGB_FOLDERS[camera])
    images = []
    for i in range(num_steps):
        img_path = os.path.join(folder, IMAGE_FORMAT % i)
        img = Image.open(img_path)
        if img.size[0] != image_size or img.size[1] != image_size:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        images.append(np.array(img)[..., :3])  # ensure RGB only
    return images


def extract_state(obs) -> np.ndarray:
    """Extract 9D state vector from an RLBench Observation.

    Returns: joint_positions (7) + gripper (2).
    If gripper_joint_positions is unavailable, falls back to
    [gripper_open * 0.04, gripper_open * 0.04] to approximate
    Franka finger joint positions (max aperture ~0.04m each).
    """
    jp = np.array(obs.joint_positions, dtype=np.float32)  # (7,)

    if obs.gripper_joint_positions is not None and len(obs.gripper_joint_positions) >= 2:
        gripper = np.array(obs.gripper_joint_positions[:2], dtype=np.float32)
    else:
        # Approximate: gripper_open is 0 (closed) or 1 (open).
        g = float(obs.gripper_open) if obs.gripper_open is not None else 0.0
        gripper = np.array([g * 0.04, g * 0.04], dtype=np.float32)

    return np.concatenate([jp, gripper])


def main():
    parser = argparse.ArgumentParser(description="Convert RLBench demos to Zarr for A2A training.")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root directory of the RLBench dataset")
    parser.add_argument("--task_name", type=str, required=True,
                        help="RLBench task name (e.g. reach_target, close_jar)")
    parser.add_argument("--num_demos", type=int, default=-1,
                        help="Number of demos to convert (-1 = all available)")
    parser.add_argument("--variation", type=int, default=0,
                        help="Task variation number")
    parser.add_argument("--camera", type=str, default="front",
                        choices=list(CAMERA_RGB_FOLDERS.keys()),
                        help="Which camera to use as head_camera")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Output image size (square)")
    parser.add_argument("--output_dir", type=str, default="data_policy",
                        help="Directory to save the zarr output")
    args = parser.parse_args()

    # Locate episodes
    episodes_dir = os.path.join(
        args.dataset_root, args.task_name,
        "variation%d" % args.variation, "episodes"
    )
    if not os.path.isdir(episodes_dir):
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")

    # Collect episode directories sorted naturally
    episode_dirs = sorted(
        [d for d in os.listdir(episodes_dir)
         if os.path.isdir(os.path.join(episodes_dir, d)) and d.startswith("episode")],
        key=lambda x: int(x.replace("episode", ""))
    )

    if args.num_demos > 0:
        episode_dirs = episode_dirs[:args.num_demos]
    num_demos = len(episode_dirs)

    print(f"Task: {args.task_name}, Variation: {args.variation}")
    print(f"Found {num_demos} episodes in {episodes_dir}")

    # Prepare output zarr (auto-increment version if path already exists)
    base_name = f"{args.task_name}_rlbench_v{args.variation}_{num_demos}"
    save_path = os.path.join(args.output_dir, f"{base_name}.zarr")
    if os.path.exists(save_path):
        version = 2
        while os.path.exists(os.path.join(args.output_dir, f"{base_name}_ver{version}.zarr")):
            version += 1
        save_path = os.path.join(args.output_dir, f"{base_name}_ver{version}.zarr")
        print(f"Output already exists. Saving as version {version}: {save_path}")

    zarr_root = zarr.group(save_path)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # Accumulate in batches
    batch_size = 100
    head_camera_list = []
    state_list = []
    action_list = []
    episode_ends_list = []
    total_frames = 0
    current_batch = 0

    for ep_idx, ep_name in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        ep_path = os.path.join(episodes_dir, ep_name)

        # Load low-dim observations
        obs_list = load_demo_low_dim(ep_path)
        num_steps = len(obs_list)

        # Load images
        images = load_demo_images(ep_path, args.camera, args.image_size, num_steps)

        # Extract states for all timesteps
        states = [extract_state(obs) for obs in obs_list]

        for t in range(num_steps):
            # Image: HWC -> CHW
            rgb = np.moveaxis(images[t], -1, 0)  # (3, H, W) uint8
            head_camera_list.append(rgb)

            state_list.append(states[t])

            # Action = target state at next timestep; last step holds position
            if t < num_steps - 1:
                action_list.append(states[t + 1])
            else:
                action_list.append(states[t])

            total_frames += 1

        episode_ends_list.append(total_frames)

        # Flush batch to zarr
        if (ep_idx + 1) % batch_size == 0 or (ep_idx + 1) == num_demos:
            hc = np.array(head_camera_list)
            st = np.array(state_list, dtype=np.float32)
            ac = np.array(action_list, dtype=np.float32)
            ee = np.array(episode_ends_list, dtype=np.int64)

            if current_batch == 0:
                zarr_data.create_dataset(
                    "head_camera", shape=(0, *hc.shape[1:]),
                    chunks=(batch_size, *hc.shape[1:]),
                    dtype=hc.dtype, compressor=compressor, overwrite=True)
                zarr_data.create_dataset(
                    "state", shape=(0, st.shape[1]),
                    chunks=(batch_size, st.shape[1]),
                    dtype="float32", compressor=compressor, overwrite=True)
                zarr_data.create_dataset(
                    "action", shape=(0, ac.shape[1]),
                    chunks=(batch_size, ac.shape[1]),
                    dtype="float32", compressor=compressor, overwrite=True)
                zarr_meta.create_dataset(
                    "episode_ends", shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64", compressor=compressor, overwrite=True)

            zarr_data["head_camera"].append(hc)
            zarr_data["state"].append(st)
            zarr_data["action"].append(ac)
            zarr_meta["episode_ends"].append(ee)

            print(f"  Batch {current_batch + 1}: wrote {len(hc)} frames")
            head_camera_list, state_list, action_list, episode_ends_list = [], [], [], []
            current_batch += 1

    # Save metadata
    meta_info = {
        "source": "rlbench",
        "task_name": args.task_name,
        "variation": args.variation,
        "camera": args.camera,
        "image_size": args.image_size,
        "num_episodes": num_demos,
        "total_frames": total_frames,
        "observation_space": "joint_pos",
        "action_space": "joint_pos",
        "state_dim": 9,
    }
    for k, v in meta_info.items():
        zarr_meta.attrs[k] = v

    import json
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(meta_info, f, indent=4)

    print(f"\nDone! Zarr saved to: {save_path}")
    print(f"  Episodes: {num_demos}, Total frames: {total_frames}")
    print(f"  head_camera shape: {zarr_data['head_camera'].shape}")
    print(f"  state shape:       {zarr_data['state'].shape}")
    print(f"  action shape:      {zarr_data['action'].shape}")


if __name__ == "__main__":
    main()
