#!/bin/bash

accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes 8 \
    ppo_training.py \
    --model_name=EleutherAI/gpt-j-6b \
    --tokenizer_name=EleutherAI/gpt-j-6b \
    --reward_model_output_path=/fsx/home-augustas/logs/unifiedqa-v2-t5-3b-1363200_custom_data_all_20230622_180051_15555 \
    --dataset_name=AugustasM/burns-ppo-training-dataset \
    --log_with=tensorboard \
    --save_freq=100 \
    --output_max_length=128 \
    --batch_size=8 \
    --gradient_accumulation_steps=8 \
    --batched_gen=True \
    --ppo_epochs=4 \
    --seed=0 \
    --learning_rate=1.4e-5 \
    --early_stopping=True \
    --output_dir=llama-se-rl-finetune-128-8-8-1.4e-5_adam
