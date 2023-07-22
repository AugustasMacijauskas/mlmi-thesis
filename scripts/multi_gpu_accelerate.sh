#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=20:00:00
#SBATCH --mail-type=NONE


# ----------------------------------------
# 
# My own code starts below
# 
# ----------------------------------------


# ----------------------------------------
# Environment configuration
# ----------------------------------------
conda env list | grep "*"
python --version
nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1


# ----------------------------------------
# Configuring GPUs
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

# Use awk to get the number of available devices
num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "Total number of GPUs: $num_gpus"


# ----------------------------------------
# Launch the job
# ----------------------------------------
workdir="$SLURM_SUBMIT_DIR"
cd $workdir

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"


# ----------------------------------------
# Model
# ----------------------------------------
model="gpt2-xl"
echo "Model: $model"


# ----------------------------------------
# Policy tokenizer
# ----------------------------------------
tokenizer="gpt2-xl"
echo "Tokenizer: $tokenizer"


# ----------------------------------------
# Save path
# ----------------------------------------
now=$(date "+%Y%m%d_%H%M%S")
model="gpt2-xl"
# keyword="${model}_unifiedqa_3b_custom_data_v4"
keyword="${model}_unifiedqa_3b_imdb"

cd ..
save_path_stem="${keyword}_${now}_${JOBID}"
save_path="ppo_logs/$save_path_stem"
mkdir $save_path
cd $workdir


# ----------------------------------------
# Reward model
# ----------------------------------------
# reward_model_output_path="/fsx/home-augustas/logs/unifiedqa-v2-t5-3b-1363200_custom_data_v4_all_20230629_120158_21789"
# reward_model_output_path="/fsx/home-augustas/logs/unifiedqa-v2-t5-11b-1363200_custom_data_imdb_v2_first_20230705_144420_27570"
# reward_model_output_path="/fsx/home-augustas/logs/unifiedqa-v2-t5-3b-1363200_custom_data_imdb_v2_first_20230707_170052_28991"
reward_model_output_path="/fsx/home-augustas/logs/UQA-3b-custom_data_imdb_v2_final_20230717_200713_36998"
echo "Reward model output path: $reward_model_output_path"


# ----------------------------------------
# Dataset
# ----------------------------------------
# dataset="AugustasM/burns-datasets-VINC-ppo-training-v3"
# dataset="AugustasM/burns-datasets-VINC-ppo-training-v4"
dataset="AugustasM/burns-datasets-VINC-imdb-ppo-training-v2"
echo -e "Dataset: $dataset\n"


# ----------------------------------------
# Template path
# ----------------------------------------
template_path="AugustasM/burns-datasets-VINC"


# ----------------------------------------
# Launch the command
# ----------------------------------------
application="accelerate"

options="launch --multi_gpu --num_machines=1 --num_processes=$num_gpus \
    --mixed_precision=no --dynamo_backend=no \
    src/ppo/ppo_training.py \
    --model_name=$model \
    --tokenizer_name=$tokenizer \
    --reward_model_output_path=$reward_model_output_path \
    --dataset_name=$dataset \
    --template_path=$template_path \
    --remove_unused_columns=False \
    --log_with=tensorboard \
    --logging_dir=/fsx/home-augustas/$save_path/ \
    --learning_rate=1e-5 \
    --batch_size=32 \
    --rm_batch_size=64 \
    --generator_batch_size=4 \
    --ppo_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --steps=384 \
    --ppo_epochs=4 \
    --early_stopping=True \
    --reward_baseline=0.0 \
    --target_kl=0.1 \
    --init_kl_coef=0.2 \
    --adap_kl_ctrl=True \
    --seed=0 \
    --save_freq=4 \
    --output_dir=/fsx/home-augustas/$save_path/checkpoints/model_ \
"

out_file_path="../$save_path/out.$JOBID"
CMD="$application $options > $out_file_path"


# ----------------------------------------
# Log the start time and execute the command
# ----------------------------------------
start=`date +%s`
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD


# ----------------------------------------
# Move the output file
# ----------------------------------------
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Log the duration
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
