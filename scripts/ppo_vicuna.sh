#!/bin/bash

# Some guidelines for good default options are:
# 1 node, 8 GPUs, 12 CPUs per GPU, 6 hours
# To get email notifications use the --mail-type option

#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J <job-name>
#SBATCH --time=06:00:00


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


# ----------------------------------------
# Configuring GPUs
# ----------------------------------------
# YOU PROBABLY DO NOT NEED TO EDIT THIS


cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

# Use awk to get the number of available devices
num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "Total number of GPUs: $num_gpus"

nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1


# ----------------------------------------
# Launch the job - DO NOT EDIT
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
model="lmsys/vicuna-7b-v1.5"
echo "Model: $model"


# ----------------------------------------
# Policy tokenizer
# ----------------------------------------
tokenizer="lmsys/vicuna-7b-v1.5"
echo "Tokenizer: $tokenizer"


# ----------------------------------------
# Save path
# ----------------------------------------
now=$(date "+%Y%m%d_%H%M%S")
keyword="vicuna-7B-UQA-3B" # Edit to your liking

save_path_stem="${keyword}_${now}_${JOBID}"
save_path="ppo_logs/$save_path_stem"
cd ..
mkdir $save_path
cd $workdir


# ----------------------------------------
# Reward model
# ----------------------------------------

# This is the path to the directory where `elk` saves the logs.
# This should be a directory that starts with `logs_elk`, but here
# it is named `logs` because I renamed the directory at some point.

# reward_model_output_path="../logs/UQA-varied-custom_data_qnli_vicuna_v1_20230721_234029_40903" # Large
reward_model_output_path="../logs/UQA-varied-custom_data_qnli_vicuna_v1_20230721_234034_40904" # 3B
echo "Reward model output path: $reward_model_output_path"


# ----------------------------------------
# Dataset
# ----------------------------------------
dataset="AugustasM/qnli-vicuna-ppo-training-v1"
echo -e "Dataset: $dataset\n"


# ----------------------------------------
# Template path
# ----------------------------------------
template_path="AugustasM/truthfulness-prompts"


# ----------------------------------------
# Launch the command
# ----------------------------------------
options="launch --multi_gpu --num_machines=1 --num_processes=$num_gpus \
    --mixed_precision=no --dynamo_backend=no \
    src/ppo/ppo_training_lora.py \
    --model_name=$model \
    --tokenizer_name=$tokenizer \
    --reward_model_output_path=$reward_model_output_path \
    --dataset_name=$dataset \
    --remove_unused_columns=False \
    --num_examples=81920 \
    --template_path=$template_path \
    --log_with=wandb \
    --logging_dir=../$save_path/ \
    --wandb_group=two_tokens \
    --learning_rate=1e-5 \
    --batch_size=16 \
    --rm_batch_size=16 \
    --generator_batch_size=16 \
    --ppo_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --steps=640 \
    --ppo_epochs=4 \
    --early_stopping=True \
    --reward_baseline=0.0 \
    --target_kl=0.1 \
    --init_kl_coef=0.2 \
    --adap_kl_ctrl=True \
    --vf_coef=1.0 \
    --seed=0 \
    --save_freq=16 \
    --output_dir=../$save_path/checkpoints/model_ \
    --log_freq=2 \
    --postprocess_responses=True \
    --full_lora=False \
"

out_file_path="../$save_path/out.$JOBID"
CMD="accelerate $options > $out_file_path"


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
