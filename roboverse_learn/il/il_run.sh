#!/bin/bash
# Usage (single task):
#   bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name a2a
#
# Usage (multi-task, '+'-separated list):
#   bash roboverse_learn/il/il_run.sh --task_name_set "close_box+stack_cube" --policy_name a2a
#
# For multi-task training, all specified tasks must use the same robot (same state/action dims).
# The eval task can be overridden with --eval_task (defaults to the first task).
export PYTHONPATH=$(pwd):$PYTHONPATH

# Force IsaacSim to exit cleanly after close (avoid shutdown hang)
export METASIM_FORCE_EXIT_ON_CLOSE=1
export METASIM_CLOSE_TIMEOUT_SEC=8

# Tasks: single name (e.g. "close_box") or '+'-separated list (e.g. "close_box+stack_cube")
task_name_set="close_box"
policy_name="a2a"    # IL policy, opts: ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, a2a, a2a_mini, a2a_reg, a2a_noise, act, score
sim_set="isaacsim"   # Simulator, e.g., mujoco, isaacsim
demo_num=100         # Number of demonstrations per task

# Training/eval control
train_enable=True
eval_enable=True
eval_task=""         # Task to evaluate on (default: first task in task_name_set)

# Training parameters
num_epochs=200
seed=42
gpu=0
num_gpus=1           # Number of GPUs (>1 enables distributed training via torchrun)
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300

# Domain Randomization Level
dr_level_collect=0
dr_level_eval=0

# Parse parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name_set)
            task_name_set="$2"
            shift 2
            ;;
        --policy_name)
            policy_name="$2"
            shift 2
            ;;
        --sim_set)
            sim_set="$2"
            shift 2
            ;;
        --demo_num)
            demo_num="$2"
            shift 2
            ;;
        --train_enable)
            train_enable="$2"
            shift 2
            ;;
        --eval_enable)
            eval_enable="$2"
            shift 2
            ;;
        --eval_task)
            eval_task="$2"
            shift 2
            ;;
        --dr_level_collect)
            dr_level_collect="$2"
            shift 2
            ;;
        --dr_level_eval)
            dr_level_eval="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --num_gpus)
            num_gpus="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Optional parameters: --task_name_set --policy_name --sim_set --demo_num"
            echo "                     --train_enable --eval_enable --eval_task"
            echo "                     --num_epochs --gpu --num_gpus --dr_level_collect --dr_level_eval"
            exit 1
            ;;
    esac
done

# Split task_name_set on '+' into an array
IFS='+' read -ra task_array <<< "${task_name_set}"
num_tasks=${#task_array[@]}

# Derive a combined name for output dirs (replace '+' with '_')
combined_task_name="${task_name_set//+/_}"

# Default eval task to the first task in the list
if [ -z "${eval_task}" ]; then
    eval_task="${task_array[0]}"
fi

# # Collect demo
# echo "=== Running collect_demo.sh ==="
# bash ./roboverse_learn/il/collect_demo.sh

# Map policy_name to model config
config_name="default_runner"
main_script="./roboverse_learn/il/train.py"

# if policy_name is ACT
if [ "${policy_name}" = "act" ]; then
    echo "=== Running ACT training and evaluation==="
    sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^train_enable=.*/train_enable=$train_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_enable=.*/eval_enable=$eval_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^collect_level=.*/collect_level=$dr_level_collect/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_level=.*/eval_level=$dr_level_eval/" ./roboverse_learn/il/policies/act/act_run.sh
    bash ./roboverse_learn/il/policies/act/act_run.sh
    echo "=== Completed all data collection, training, and evaluation ==="
    exit 0
fi

# Run training/evaluation for DP/FM/VITA/A2A policies
echo "=== Running ${policy_name} on task(s): ${task_name_set} ==="

# Launcher: use torchrun for multi-GPU, plain python for single-GPU
if [ "${num_gpus}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${num_gpus}"
else
    LAUNCHER="python"
fi

eval_ckpt_name=$num_epochs
output_dir="./il_outputs/${policy_name}"
eval_path="${output_dir}/${combined_task_name}/checkpoints/${eval_ckpt_name}.ckpt"

echo "Checkpoint path: $eval_path"

extra="obs:${obs_space}_act:${act_space}"
if [ "${delta_ee}" = 1 ]; then
  extra="${extra}_delta"
fi

export policy_name="${policy_name}"

if [ "${num_tasks}" -eq 1 ]; then
    # ---- Single-task training (original behaviour) ----
    zarr_path="./data_policy/${task_name_set}FrankaL${dr_level_collect}_${extra}_${demo_num}.zarr"

    ${LAUNCHER} ${main_script} --config-name=${config_name}.yaml \
    task_name=${combined_task_name} \
    "dataset_config.zarr_path=${zarr_path}" \
    train_config.training_params.seed=${seed} \
    train_config.training_params.num_epochs=${num_epochs} \
    train_config.training_params.device=${gpu} \
    eval_config.policy_runner.obs.obs_type=${obs_space} \
    eval_config.policy_runner.action.action_type=${act_space} \
    eval_config.policy_runner.action.delta=${delta_ee} \
    eval_config.eval_args.task=${eval_task} \
    eval_config.eval_args.max_step=${eval_max_step} \
    eval_config.eval_args.num_envs=${eval_num_envs} \
    eval_config.eval_args.sim=${sim_set} \
    eval_config.eval_args.level=${dr_level_eval} \
    +eval_config.eval_args.max_demo=50 \
    train_enable=${train_enable} \
    eval_enable=${eval_enable} \
    eval_path=${eval_path}
else
    # ---- Multi-task training ----
    # Build the hydra list of zarr paths: [path1,path2,...]
    zarr_paths_list=""
    for task in "${task_array[@]}"; do
        path="./data_policy/${task}FrankaL${dr_level_collect}_${extra}_${demo_num}.zarr"
        if [ -z "${zarr_paths_list}" ]; then
            zarr_paths_list="${path}"
        else
            zarr_paths_list="${zarr_paths_list},${path}"
        fi
    done

    # Build task_descriptions dict for task embedding: {task_name: "human readable description"}
    task_descs_entries=""
    for task in "${task_array[@]}"; do
        desc="${task//_/ }"
        if [ -z "${task_descs_entries}" ]; then
            task_descs_entries="${task}: '${desc}'"
        else
            task_descs_entries="${task_descs_entries}, ${task}: '${desc}'"
        fi
    done

    ${LAUNCHER} ${main_script} --config-name=${config_name}.yaml \
    dataset_config=multi_task_robot_image_dataset \
    task_name=${combined_task_name} \
    "dataset_config.zarr_paths=[${zarr_paths_list}]" \
    "+policy_config.task_descriptions={${task_descs_entries}}" \
    train_config.training_params.seed=${seed} \
    train_config.training_params.num_epochs=${num_epochs} \
    train_config.training_params.device=${gpu} \
    eval_config.policy_runner.obs.obs_type=${obs_space} \
    eval_config.policy_runner.action.action_type=${act_space} \
    eval_config.policy_runner.action.delta=${delta_ee} \
    eval_config.eval_args.task=${eval_task} \
    eval_config.eval_args.max_step=${eval_max_step} \
    eval_config.eval_args.num_envs=${eval_num_envs} \
    eval_config.eval_args.sim=${sim_set} \
    eval_config.eval_args.level=${dr_level_eval} \
    +eval_config.eval_args.max_demo=50 \
    train_enable=${train_enable} \
    eval_enable=${eval_enable} \
    eval_path=${eval_path}
fi

echo "=== Completed all data collection, training, and evaluation ==="
