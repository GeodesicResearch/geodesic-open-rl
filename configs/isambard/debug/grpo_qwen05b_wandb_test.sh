#!/bin/bash
# Wandb tracking test: 0.5B model, ~100 training steps, 2 nodes.
#
# Usage: sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_qwen05b_wandb_test.sh

# Disable job chaining for test runs
export MAX_JOB_CHAINS=0

EXP_NAME="${EXP_NAME:-grpo_qwen05b_wandb_test}"

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
    --total_episodes 1600 \
    --deepspeed_stage 0 \
    --num_nodes 2 \
    --num_learners_per_node 1 \
    --vllm_num_engines 3 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --dataset_mixer_eval_list ai2-adapt-dev/math_ground_truth_zs 1.0 \
    --dataset_mixer_eval_list_splits test \
    --seed 42 \
    --save_freq 25 \
    --checkpoint_state_freq 25 \
    --checkpoint_state_dir /projects/a5k/public/checkpoints_${USER}/grpo-rlzero/${EXP_NAME} \
    --push_to_hub false \
    --with_tracking true \
    --output_dir /projects/a5k/public/models_${USER}/grpo-rlzero/${EXP_NAME}/checkpoints \
"
