set -x

export RAY_TMPDIR="/workspace/verl_exp/"
export WANDB_API_KEY=37f371d2968f35d69749ee52089583eb8e1f0cab
export WANDB_DIR="/workspace/verl_exp/"
export WANDB_MODE=offline
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="6,7"

project_name='GRPO'
exp_name='GRPO-Qwen2.5-1.5B'

adv_estimator=grpo

use_kl_in_reward=False
use_kl_loss=True
kl_loss_coef=0.005

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 4))

train_prompt_bsz=128
n_resp_per_prompt=8
train_prompt_mini_bsz=128
filter_overlong_prompts=True

loss_agg_mode="token-mean"

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/workspace/verl_exp"}
MODEL_PATH=${MODEL_PATH:-"/models/Qwen/Qwen2.5-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/math/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/math/test.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1  # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

offload=True

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=${filter_overlong_prompts} \
    data.truncation='error' \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes="${NNODES}" \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.total_epochs=1 2>&1 | tee grpo_qwen2.5_1.5b.log