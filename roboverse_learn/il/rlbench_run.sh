#!/bin/bash
# =============================================================================
# Train and evaluate A2A policy on RLBench tasks.
#
# This script converts RLBench demos to Zarr format, trains the A2A policy
# using the existing training pipeline, and evaluates on RLBench's CoppeliaSim.
#
# Usage:
#   bash roboverse_learn/il/rlbench_run.sh \
#       --task_name reach_target \
#       --dataset_root /path/to/rlbench_data
#
# Multi-task:
#   bash roboverse_learn/il/rlbench_run.sh \
#       --task_name "reach_target+close_jar" \
#       --dataset_root /path/to/rlbench_data
# =============================================================================
export PYTHONPATH=$(pwd):$PYTHONPATH

# ---- Default parameters ----
task_name="reach_target"
dataset_root=""           # RLBench dataset root (required)
policy_name="a2a"
num_demos=100             # Demos per task to convert
variation=0
camera="front"
image_size=256

# Training control
train_enable=True
eval_enable=True
num_epochs=200
seed=42
gpu=0

# Eval control
eval_num_episodes=25
eval_max_steps=200
eval_num_workers=1
headless="--headless"     # Remove to show GUI

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name)        task_name="$2"; shift 2 ;;
        --dataset_root)     dataset_root="$2"; shift 2 ;;
        --policy_name)      policy_name="$2"; shift 2 ;;
        --num_demos)        num_demos="$2"; shift 2 ;;
        --variation)        variation="$2"; shift 2 ;;
        --camera)           camera="$2"; shift 2 ;;
        --image_size)       image_size="$2"; shift 2 ;;
        --train_enable)     train_enable="$2"; shift 2 ;;
        --eval_enable)      eval_enable="$2"; shift 2 ;;
        --num_epochs)       num_epochs="$2"; shift 2 ;;
        --gpu)              gpu="$2"; shift 2 ;;
        --eval_num_episodes) eval_num_episodes="$2"; shift 2 ;;
        --eval_max_steps)   eval_max_steps="$2"; shift 2 ;;
        --eval_num_workers) eval_num_workers="$2"; shift 2 ;;
        --no-headless)      headless=""; shift ;;
        *)
            echo "Unknown parameter: $1"
            exit 1 ;;
    esac
done

if [ -z "${dataset_root}" ]; then
    echo "ERROR: --dataset_root is required (path to RLBench dataset)"
    exit 1
fi

# ---- Split tasks on '+' ----
IFS='+' read -ra task_array <<< "${task_name}"
num_tasks=${#task_array[@]}
combined_name="${task_name//+/_}"

echo "=== RLBench A2A Pipeline ==="
echo "Tasks: ${task_name} (${num_tasks} task(s))"
echo "Dataset root: ${dataset_root}"
echo "Policy: ${policy_name}"

# =============================================================================
# Step 1: Convert RLBench demos to Zarr
# =============================================================================
echo ""
echo "=== Step 1: Converting RLBench demos to Zarr ==="

zarr_paths_list=""
for task in "${task_array[@]}"; do
    zarr_path="./data_policy/${task}_rlbench_v${variation}_${num_demos}.zarr"

    if [ -d "${zarr_path}" ]; then
        echo "  Zarr already exists: ${zarr_path} (skipping conversion)"
    else
        echo "  Converting: ${task} -> ${zarr_path}"
        python roboverse_learn/il/rlbench2zarr.py \
            --dataset_root "${dataset_root}" \
            --task_name "${task}" \
            --num_demos "${num_demos}" \
            --variation "${variation}" \
            --camera "${camera}" \
            --image_size "${image_size}" \
            --output_dir data_policy

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to convert ${task}"
            exit 1
        fi
    fi

    if [ -z "${zarr_paths_list}" ]; then
        zarr_paths_list="${zarr_path}"
    else
        zarr_paths_list="${zarr_paths_list},${zarr_path}"
    fi
done

# =============================================================================
# Step 2: Train A2A policy
# =============================================================================
config_name="default_runner"
main_script="./roboverse_learn/il/train.py"
output_dir="./il_outputs/${policy_name}"
eval_path="${output_dir}/${combined_name}/checkpoints/${num_epochs}.ckpt"

export policy_name="${policy_name}"

if [ "${train_enable}" = "True" ]; then
    echo ""
    echo "=== Step 2: Training ${policy_name} on ${task_name} ==="

    if [ "${num_tasks}" -eq 1 ]; then
        zarr_path="./data_policy/${task_array[0]}_rlbench_v${variation}_${num_demos}.zarr"

        python ${main_script} --config-name=${config_name}.yaml \
            task_name=${combined_name} \
            "dataset_config.zarr_path=${zarr_path}" \
            train_config.training_params.seed=${seed} \
            train_config.training_params.num_epochs=${num_epochs} \
            train_config.training_params.device=${gpu} \
            eval_config.policy_runner.obs.obs_type=joint_pos \
            eval_config.policy_runner.action.action_type=joint_pos \
            eval_config.policy_runner.action.delta=0 \
            train_enable=True \
            eval_enable=False \
            eval_path=${eval_path}
    else
        python ${main_script} --config-name=${config_name}.yaml \
            dataset_config=multi_task_robot_image_dataset \
            task_name=${combined_name} \
            "dataset_config.zarr_paths=[${zarr_paths_list}]" \
            train_config.training_params.seed=${seed} \
            train_config.training_params.num_epochs=${num_epochs} \
            train_config.training_params.device=${gpu} \
            eval_config.policy_runner.obs.obs_type=joint_pos \
            eval_config.policy_runner.action.action_type=joint_pos \
            eval_config.policy_runner.action.delta=0 \
            train_enable=True \
            eval_enable=False \
            eval_path=${eval_path}
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed"
        exit 1
    fi
else
    echo ""
    echo "=== Step 2: Training skipped (train_enable=False) ==="
fi

# =============================================================================
# Step 3: Evaluate on RLBench (CoppeliaSim)
# =============================================================================
if [ "${eval_enable}" = "True" ]; then
    echo ""
    echo "=== Step 3: Evaluating on RLBench ==="

    eval_task="${task_array[0]}"

    if [ ! -f "${eval_path}" ]; then
        echo "ERROR: Checkpoint not found at ${eval_path}"
        exit 1
    fi

    python roboverse_learn/il/rlbench_eval.py \
        --checkpoint "${eval_path}" \
        --task_name "${eval_task}" \
        --num_episodes "${eval_num_episodes}" \
        --max_steps "${eval_max_steps}" \
        --camera "${camera}" \
        --image_size "${image_size}" \
        --variation "${variation}" \
        --device "cuda:${gpu}" \
        --dataset_root "${dataset_root}" \
        --num_workers "${eval_num_workers}" \
        ${headless}
else
    echo ""
    echo "=== Step 3: Evaluation skipped (eval_enable=False) ==="
fi

echo ""
echo "=== RLBench A2A Pipeline Complete ==="
