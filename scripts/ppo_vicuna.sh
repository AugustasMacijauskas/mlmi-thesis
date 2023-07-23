#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=24:00:00
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
model="lmsys/vicuna-7b-v1.3"
echo "Model: $model"


# ----------------------------------------
# Policy tokenizer
# ----------------------------------------
tokenizer="huggyllama/llama-7b"
echo "Tokenizer: $tokenizer"


# ----------------------------------------
# Save path
# ----------------------------------------
now=$(date "+%Y%m%d_%H%M%S")
keyword="vicuna_UQA_3b_qnli"

cd ..
save_path_stem="${keyword}_${now}_${JOBID}"
save_path="ppo_logs/$save_path_stem"
mkdir $save_path
cd $workdir


# ----------------------------------------
# Reward model
# ----------------------------------------

# ---------------------------------------- QNLI Vicuna ----------------------------------------
# reward_model_output_path="/fsx/home-augustas/logs/UQA-varied-custom_data_qnli_vicuna_v1_20230721_234029_40903" # Large
reward_model_output_path="/fsx/home-augustas/logs/UQA-varied-custom_data_qnli_vicuna_v1_20230721_234034_40904" # 3B
echo "Reward model output path: $reward_model_output_path"


# ----------------------------------------
# Dataset
# ----------------------------------------
dataset="AugustasM/qnli-vicuna-ppo-training-v1"
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
    src/ppo/ppo_training_lora.py \
    --model_name=$model \
    --tokenizer_name=$tokenizer \
    --reward_model_output_path=$reward_model_output_path \
    --dataset_name=$dataset \
    --template_path=$template_path \
    --remove_unused_columns=False \
    --log_with=wandb \
    --logging_dir=/fsx/home-augustas/$save_path/ \
    --wandb_group=vf_coef_tests \
    --learning_rate=1e-6 \
    --batch_size=128 \
    --rm_batch_size=64 \
    --generator_batch_size=16 \
    --ppo_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --steps=80 \
    --ppo_epochs=4 \
    --early_stopping=True \
    --reward_baseline=0.0 \
    --target_kl=0.1 \
    --init_kl_coef=0.2 \
    --adap_kl_ctrl=True \
    --vf_coef=2.0 \
    --seed=0 \
    --save_freq=2 \
    --output_dir=/fsx/home-augustas/$save_path/checkpoints/model_ \
    --is_lora=True \
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
