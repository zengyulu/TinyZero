#!/bin/bash

export HYDRA_FULL_ERROR=1
export N_GPUS=2
export BASE_MODEL=/workspace/Qwen2.5-1.5B
export DATA_DIR=/workspace/TinyZero/data/countdown
export ROLLOUT_TP_SIZE=1

wandb login e447ec43d0319f54f231ffd93512571cee4e2162

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=64 \
  data.val_batch_size=128 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_micro_batch_size=2 \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.grad_offload=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=local_countdown-qwen2.5-1.5b \
  trainer.total_epochs=15 2>&1 | tee verl_demo.log