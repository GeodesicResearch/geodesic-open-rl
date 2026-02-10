#!/bin/bash
# Training config for OLMo3-7B GRPO RL-Zero (general mix) on Isambard.
# Adapted from scripts/train/olmo3/7b_rlzero_general.sh for 4-GPU/node topology.
#
# Usage: sbatch configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_7b_rlzero_general.sh
#        sbatch --nodes=10 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_7b_rlzero_general.sh
#
# GPU topology (10 nodes x 4 GPUs = 40 GPUs total):
#   - 20 learner GPUs: 10 nodes x 2 GPUs/node (num_learners_per_node=2)
#   - 20 vLLM engines: remaining GPUs for inference

# Model — OLMo3-7B base from HuggingFace
# TODO: replace with actual OLMo3-7B base model ID when ready to step up
MODEL_NAME_OR_PATH="allenai/OLMo-2-0325-7B"

DATASETS="hamishivi/rlvr_general_mix 13314"

LOCAL_EVALS="hamishivi/rlvr_general_mix 8"
LOCAL_EVAL_SPLITS="train"

EXP_NAME="${EXP_NAME:-grpo_general_from_zero_isambard}"

TRAINING_ARGS="\
    --exp_name ${EXP_NAME} \
    --beta 0.0 \
    --async_steps 4 \
    --inflight_updates \
    --truncated_importance_sampling_ratio_cap 2.0 \
    --num_samples_per_prompt_rollout 8 \
    --num_unique_prompts_rollout 32 \
    --num_mini_batches 1 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --kl_estimator 2 \
    --dataset_mixer_list $DATASETS \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list $LOCAL_EVALS \
    --dataset_mixer_eval_list_splits $LOCAL_EVAL_SPLITS \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 16384 \
    --pack_length 18432 \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --chat_template_name olmo_thinker \
    --stop_strings '</answer>' \
    --non_stop_penalty False \
    --temperature 1.0 \
    --total_episodes 10000000 \
    --deepspeed_stage 3 \
    --num_learners_per_node 2 \
    --vllm_num_engines 20 \
    --vllm_tensor_parallel_size 1 \
    --vllm_sync_backend gloo \
    --llm_judge_model null/null \
    --llm_judge_timeout 600 \
    --llm_judge_max_tokens 2048 \
    --llm_judge_max_context_length 32768 \
    --lr_scheduler_type constant \
    --apply_verifiable_reward true \
    --seed 1 \
    --local_eval_every 50 \
    --save_freq 50 \
    --checkpoint_state_freq 50 \
    --checkpoint_state_dir /projects/a5k/public/checkpoints_${USER}/grpo-rlzero/${EXP_NAME} \
    --gradient_checkpointing \
    --with_tracking \
    --vllm_enable_prefix_caching \
    --clip_higher 0.272 \
    --keep_last_n_checkpoints -1 \
    --mask_truncated_completions True \
    --push_to_hub false \
    --output_dir /projects/a5k/public/models_${USER}/grpo-rlzero/${EXP_NAME}/checkpoints \
"
