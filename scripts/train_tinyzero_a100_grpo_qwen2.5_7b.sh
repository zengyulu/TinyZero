#!/bin/bash

export HYDRA_FULL_ERROR=1
export N_GPUS=2
export BASE_MODEL=/workspace/Qwen2.5-7B
export DATA_DIR=/workspace/TinyZero/data/countdown
export ROLLOUT_TP_SIZE=1  #7B 模型单卡可容纳，关闭张量并行减少通信开销，优先数据并行
export VLLM_ATTENTION_BACKEND=FLASH_ATTENTION_2 # 使用FLASH ATTENTION v2

wandb login e447ec43d0319f54f231ffd93512571cee4e2162

# learning_rate: 减少学习率到 critic.optim.lr: 5e-6 actor.optim.lr:5e-7
# ppo_mini_batch_size: 增加到 128
# ppo_micro_batch_size: 增加到 8
# train和val的 全局batch_size 减少到128和512
# max response length 略微增加到1280
# algorithm.kl_coef 增加到 0.02
# 添加 +trainer.precision=bf16 混合精度16
#  actor_rollout_ref.model.torch_dtype=bf16
# algorithm.entropy_coef=0.02
# gpu_memory_utilization 调整到 0.6
# param_offload 全都设为False，减少向CPU传输的延迟


python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  data.train_batch_size=128 \
  data.val_batch_size=512 \
  data.max_prompt_length=256 \
  data.max_response_length=1280 \
  actor_rollout_ref.model.path=$BASE_MODEL \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.optim.lr=5e-6 \
  critic.model.path=$BASE_MODEL \
  critic.ppo_micro_batch_size=8 \
  critic.model.enable_gradient_checkpointing=True \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.grad_offload=True \
  algorithm.kl_ctrl.kl_coef=0.02 \
  trainer.critic_warmup=1000 \
  trainer.logger=['wandb'] \
  +trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=$N_GPUS \
  trainer.nnodes=1 \
  trainer.save_freq=30 \
  trainer.test_freq=30 \
  trainer.project_name=TinyZero \
  trainer.experiment_name=local_countdown-qwen2.5-7b \
  trainer.total_epochs=15 2>&1 | tee verl_demo.log \
  +trainer.precision=bf16 \
  actor_rollout_ref.model.torch_dtype=bf16 \
  algorithm.entropy_coef=0.02
