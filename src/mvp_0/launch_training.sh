#!/bin/bash

conda env list | grep "*"
python --version

# accelerate launch --multi_gpu --num_machines=1 --num_processes=1 ppo_training.py \
accelerate launch --num_processes=1 src/mvp_0/ppo_training.py \
    --model_name=gpt2 \
    --tokenizer_name=gpt2 \
    --reward_model_output_path=/fsx/home-augustas/logs/unifiedqa-v2-t5-3b-1363200_custom_data_v4_all_20230629_120158_21789 \
    --dataset_name=AugustasM/burns-datasets-VINC-ppo-training-v4 \
    --remove_unused_columns=False \
    --log_with=tensorboard \
    --logging_dir=/fsx/home-augustas/ppo_logs/test/ \
    --learning_rate=1.4e-5 \
    --batch_size=1 \
    --mini_batch_size=1 \
    --gradient_accumulation_steps=64 \
    --steps=192 \
    --ppo_epochs=4 \
    --early_stopping=True \
    --reward_baseline=0.0 \
    --target_kl=0.1 \
    --init_kl_coef=0.2 \
    --adap_kl_ctrl=True \
    --seed=0 \
    --save_freq=2 \
    --output_dir=/fsx/home-augustas/ppo_runs/test \

