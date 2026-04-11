import copy
from typing import Dict, List

import numba
import numpy as np
import torch
from roboverse_learn.il.utils.normalize_util import get_image_range_normalizer
from roboverse_learn.il.utils.pytorch_util import dict_apply
from roboverse_learn.il.utils.replay_buffer import ReplayBuffer
from roboverse_learn.il.utils.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from roboverse_learn.il.datasets.base_dataset import BaseImageDataset
from roboverse_learn.il.utils.normalizer import LinearNormalizer


class RobotImageDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
    ):

        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action"],
        )

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255.0

        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            # print(idx, len(idx))
            # print(self.batch_size)
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        return {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
            },
            "action": action,  # B, T, D
        }


def _merge_replay_buffers(buffers: List[ReplayBuffer]) -> ReplayBuffer:
    """Merge multiple ReplayBuffer instances into a single in-memory numpy buffer.

    Concatenates data arrays and adjusts episode_ends offsets accordingly.
    All buffers must share the same data keys and compatible array shapes.
    Also creates a ``task_idx`` array so each timestep knows which source task it belongs to.
    """
    keys = list(buffers[0].keys())

    # Concatenate data arrays along the time dimension
    merged_data = {
        key: np.concatenate([rb[key] for rb in buffers], axis=0)
        for key in keys
    }

    # Build per-timestep task index (int64 scalar per frame)
    task_idx_parts = []
    for i, rb in enumerate(buffers):
        task_idx_parts.append(np.full((rb.n_steps,), i, dtype=np.int64))
    merged_data["task_idx"] = np.concatenate(task_idx_parts, axis=0)

    # Merge episode_ends: shift each buffer's ends by the cumulative frame count
    all_episode_ends = []
    offset = 0
    for rb in buffers:
        ends = rb.episode_ends[:]
        all_episode_ends.append(ends + offset)
        offset += rb.n_steps  # total frames in this buffer

    merged_meta = {"episode_ends": np.concatenate(all_episode_ends)}
    return ReplayBuffer(root={"meta": merged_meta, "data": merged_data})


class MultiTaskRobotImageDataset(BaseImageDataset):
    """Dataset that merges multiple single-task zarr files for multi-task training.

    All zarr files must share the same robot configuration (identical state and
    action dimensions). An error is raised if incompatible datasets are provided,
    because mixing robots with different DOF would corrupt the training signal.
    """

    def __init__(
        self,
        zarr_paths: List[str],
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
    ):
        super().__init__()

        if len(zarr_paths) == 0:
            raise ValueError("zarr_paths must contain at least one path.")

        # Load all replay buffers
        buffers = [
            ReplayBuffer.copy_from_path(p, keys=["head_camera", "state", "action"])
            for p in zarr_paths
        ]

        # Validate that all datasets share the same robot/camera configuration
        ref_path = zarr_paths[0]
        ref_state_shape = buffers[0]["state"].shape[1:]
        ref_action_shape = buffers[0]["action"].shape[1:]
        ref_cam_shape = buffers[0]["head_camera"].shape[1:]

        for i, (rb, path) in enumerate(zip(buffers[1:], zarr_paths[1:]), 1):
            state_shape = rb["state"].shape[1:]
            action_shape = rb["action"].shape[1:]
            cam_shape = rb["head_camera"].shape[1:]

            if state_shape != ref_state_shape:
                raise ValueError(
                    f"Incompatible robot: state shape {state_shape} in '{path}' "
                    f"does not match {ref_state_shape} in '{ref_path}'. "
                    "Only mix datasets from the same robot type."
                )
            if action_shape != ref_action_shape:
                raise ValueError(
                    f"Incompatible robot: action shape {action_shape} in '{path}' "
                    f"does not match {ref_action_shape} in '{ref_path}'. "
                    "Only mix datasets from the same robot type."
                )
            if cam_shape != ref_cam_shape:
                raise ValueError(
                    f"Incompatible camera: image shape {cam_shape} in '{path}' "
                    f"does not match {ref_cam_shape} in '{ref_path}'."
                )

        # Merge all buffers into one
        self.replay_buffer = _merge_replay_buffers(buffers)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.batch_size = batch_size

        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["head_cam"] = get_image_range_normalizer()
        normalizer["front_cam"] = get_image_range_normalizer()
        normalizer["left_cam"] = get_image_range_normalizer()
        normalizer["right_cam"] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255.0
        return {
            "obs": {
                "head_cam": head_cam,
                "agent_pos": agent_pos,
            },
            "action": sample["action"].astype(np.float32),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True)
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0
        action = samples["action"].to(device, non_blocking=True)
        result = {
            "obs": {
                "head_cam": head_cam,
                "agent_pos": agent_pos,
            },
            "action": action,
        }
        if "task_idx" in samples:
            result["task_idx"] = samples["task_idx"].to(device, non_blocking=True)
        return result


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
