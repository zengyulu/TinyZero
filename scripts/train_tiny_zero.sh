#!/bin/bash

export HYDRA_FULL_ERROR=1
export N_GPUS=1
export BASE_MODEL=/workspace/Qwen2.5-7B
export DATA_DIR=/workspace/TinyZero/data/countdown
export ROLLOUT_TP_SIZE=1  #7B 模型单卡可容纳，关闭张量并行减少通信开销，优先数据并行
export VLLM_ATTENTION_BACKEND=XFORMERS # 使用FLASH ATTENTION v2

wandb login e447ec43d0319f54f231ffd93512571cee4e2162

python3 -m verl.trainer.main_ppo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=256 \
  data.val_batch_size=1312 \
  data.max_prompt_length=256 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  critic.optim.lr=1e-5 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_micro_batch_size=8 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=40 \
  trainer.test_freq=40 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=$EXPERIMENT_NAME \
  trainer.total_epochs=15 2>&1 | tee verl_demo.log