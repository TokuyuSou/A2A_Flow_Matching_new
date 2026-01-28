# A2A: Action-to-Action Flow Matching Policy

A2A is a flow matching policy that directly transforms history action distributions to future action distributions, conditioned on visual observations. This repository provides the implementation of A2A algorithm built on top of the [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse) platform.

<p align="center">
  <img src="docs/source/_static/framework.png" width="80%" alt="A2A Framework">
</p>

## Overview

A2A (Action-to-Action) uses flow matching to learn a transformation from history states to future actions, enabling efficient and robust imitation learning for robotic manipulation tasks.


### Architecture

```
History States [s_{t-n+1}, ..., s_t] --encode--> history_latents (x_0)
Visual Obs [img_{t-n+1}, ..., img_t] --encode--> obs_latents (condition)

Flow Matching: x_0 --flow(condition)--> x_1 (future_action_latents)

x_1 --decode--> Future Actions [a_t, a_{t+1}, ..., a_{t+k}]
```
<p align="center">
  <img src="docs/source/_static/pipeline.png" width="50%" alt="A2A Pipeline">
</p>

### Variants

| Policy | Description |
|--------|-------------|
| `a2a` | Base Action-to-Action flow matching policy |
| `a2a_noise` | Adds Gaussian noise to history actions for improved robustness |

## Installation

### Prerequisites

This project is built on [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse). Please follow the RoboVerse documentation for environment setup:

1. **Create Conda Environment**
   
   Please refer to the [RoboVerse Documentation](https://roboverse.wiki/metasim/) for detailed installation instructions.

2. **Install Simulators**
   
   A2A supports multiple simulators. Install the ones you need:
   
   - **Isaac Sim V5.0.0**: Follow [Isaac Sim Installation Guide](https://roboverse.wiki/metasim/get_started/installation)
   - **MuJoCo**: Follow [MuJoCo Installation Guide](https://roboverse.wiki/metasim/get_started/installation)

3. **Install A2A Dependencies**

   ```bash
   cd roboverse_learn/il/policies/a2a
   pip install -r requirements.txt
   ```

4. **Fix Potential Issues (Optional)**
   
   If you encounter issues with zarr or hydra, run the setup script to fix them:
   
   ```bash
   bash roboverse_learn/il/il_setup.sh
   ```

5. **Setup Weights & Biases (Optional)**
   
   Create a [Weights & Biases](https://wandb.ai/) account to obtain an API key for experiment logging.

## Quick Start

### 1. Collect Demonstration Data

```bash
bash roboverse_learn/il/collect_demo.sh
```

### 2. Train A2A Policy

```bash
bash roboverse_learn/il/il_run.sh \
    --task_name_set close_box \
    --policy_name a2a \
    --train_enable True \
    --eval_enable False
```

### 3. Evaluate A2A Policy

```bash
bash roboverse_learn/il/il_run.sh \
    --task_name_set close_box \
    --policy_name a2a \
    --train_enable False \
    --eval_enable True
```

## Usage

### Command Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task_name_set` | `close_box` | Task name (e.g., `close_box`, `stack_cube`, `pick_cube`) |
| `--policy_name` | `a2a` | Policy type (`a2a`, `a2a_noise`) |
| `--sim_set` | `isaacsim` | Simulator (`isaacsim`, `mujoco`) |
| `--demo_num` | `100` | Number of demonstrations |
| `--train_enable` | `True` | Enable training |
| `--eval_enable` | `True` | Enable evaluation |
| `--num_epochs` | `200` | Number of training epochs |
| `--gpu` | `0` | GPU device ID |
| `--dr_level_collect` | `0` | Domain randomization level for data collection |
| `--dr_level_eval` | `0` | Domain randomization level for evaluation |

### Examples

**Train A2A on stack_cube task with Issac Sim:**
```bash
bash roboverse_learn/il/il_run.sh \
    --task_name_set stack_cube \
    --policy_name a2a \
    --sim_set isaacsim \
    --num_epochs 300 \
    --gpu 0
```

**Evaluate with domain randomization:**
```bash
bash roboverse_learn/il/il_run.sh \
    --task_name_set close_box \
    --policy_name a2a \
    --train_enable False \
    --eval_enable True \
    --dr_level_eval 2
```

## Project Structure

```
├── roboverse_learn/
│   └── il/
│       ├── il_run.sh              # Main training/evaluation script
│       ├── collect_demo.sh        # Demo collection script
│       ├── train.py               # Training entry point
│       ├── configs/               # Hydra configuration files
│       │   └── policy_config/
│       │       ├── a2a.yaml       # A2A policy config
│       │       └── a2a_noise.yaml # A2A-Noise policy config
│       └── policies/
│           └── a2a/               # A2A policy implementation
│               ├── a2a_policy.py
│               ├── a2a_noise_policy.py
│               ├── action_ae.py
│               └── README.md
├── data_policy/                   # Collected demonstration data
└── il_outputs/                    # Training outputs and checkpoints
```

## Checkpoints

Trained model checkpoints are saved at:
```
./il_outputs/{policy_name}/{task_name}/checkpoints/{epoch}.ckpt
```

## Supported Tasks

The following tasks from RoboVerse are supported:

- `close_box`
- `stack_cube`
- `pick_cube`
- And more tasks from RoboVerse...

Please refer to [RoboVerse Documentation](https://roboverse.wiki/) for the complete list of available tasks.

## License

This project is licensed under the Apache License 2.0.

## Acknowledgments

This project is built on top of [RoboVerse](https://github.com/RoboVerseOrg/RoboVerse). We thank the RoboVerse team for providing an excellent platform for robot learning research.

## Citation

If you find A2A useful in your research, please consider citing:

```bibtex
@misc{a2a2025,
    title={A2A: Action-to-Action Flow Matching Policy},
    year={2025},
}
```

