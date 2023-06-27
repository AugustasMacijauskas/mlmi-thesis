#!/bin/bash

python src/mvp_0/ppo_training.py \
    --model_name=EleutherAI/gpt-j-6b \
    --tokenizer_name=EleutherAI/gpt-j-6b \
    --reward_model_output_path=/fsx/home-augustas/logs_old/unifiedqa-v2-t5-3b-1363200_custom_data_all_20230622_180051_15555 \
    # --reward_model_output_path=/fsx/home-augustas/logs_old/unifiedqa-v2-t5-11b-1363200_custom_data_all_20230626_154115_18944 \
    --reward_model_tokenizer_name=allenai/unifiedqa-v2-t5-3b-1363200 \
    # --reward_model_tokenizer_name=allenai/unifiedqa-v2-t5-11b-1363200 \
    --dataset_name=AugustasM/burns-ppo-training-dataset \
    # --log_with=tensorboard \
    --save_freq=100 \
    --batch_size=8 \
    --gradient_accumulation_steps=8 \
    --ppo_epochs=4 \
    --seed=0 \
    --learning_rate=1.4e-5 \
    --early_stopping=True \
    --output_dir=logs/ppo_training
