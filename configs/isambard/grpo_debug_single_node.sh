#!/bin/bash
# Debug training config for pipeline validation on a single Isambard node (4 GPUs).
# Uses a small model and short run to verify the full GRPO stack works.
#
# Usage: sbatch --nodes=1 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_debug_single_node.sh

# Disable job chaining for debug runs
export MAX_JOB_CHAINS=0

EXP_NAME="${EXP_NAME:-grpo_debug_single_node}"

TRAINING_ARGS="\
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --num_samples_per_prompt_rollout 4 \
    --num_unique_prompts_rollout 4 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --dataset_mixer_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_list_splits train \
    --max_prompt_token_length 1024 \
    --response_length 2048 \
    --pack_length 3072 \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --chat_template_name qwen \
    --temperature 1.0 \
    --total_episodes 100 \
    --deepspeed_stage 0 \
    --num_learners_per_node 1 \
    --vllm_num_engines 3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --dataset_mixer_eval_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_eval_list_splits test \
    --seed 42 \
    --save_freq 10 \
    --checkpoint_state_freq 3 \
    --checkpoint_state_dir /projects/a5k/public/checkpoints_${USER}/grpo-rlzero/${EXP_NAME} \
    --push_to_hub false \
    --with_tracking false \
    --output_dir /projects/a5k/public/models_${USER}/grpo-rlzero/${EXP_NAME}/checkpoints \
"
