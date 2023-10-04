#!/bin/bash

# Some guidelines for good default options are:
# 1 node, 8 GPUs, 12 CPUs per GPU, 3 hours
# To get email notifications use the --mail-type option

#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J <job-name>
#SBATCH --time=03:00:00


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

num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "num_gpus: $num_gpus"

export NUMEXPR_MAX_THREADS="$((num_gpus * 12))"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"

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
# Save path
# ----------------------------------------
now=$(date "+%Y%m%d_%H%M%S")
keyword="vicuna-7B"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_burns"
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
model="lmsys/vicuna-7b-v1.5"
echo "Model: $model"


# ----------------------------------------
# Tokenizer
# ----------------------------------------

# This same tokenizer as the model is used by default,
# so only change this if you want to use a different one.

# tokenizer=""
echo "Tokenizer: $tokenizer"


# ----------------------------------------
# LoRA weight path
# ----------------------------------------
lora_path="~/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_64"
# lora_path=""
echo "LoRA path: $lora_path"


# ----------------------------------------
# The task
# ----------------------------------------
task="qnli_vicuna"
# task="qnli_custom_2" # The default QNLI template wrapped with lmsys chat prefix and suffix
echo -e "Task: $task\n"


# ----------------------------------------
# Log the start time,
# cd in to the correct and
# start executing the commands
# ----------------------------------------
start=`date +%s`
cd ../lm_evaluation_harness_refactored/lm-evaluation-harness


# ----------------------------------------
# Launch the commands
# ----------------------------------------

# The batch size of 8 is for a 7B model, decrease if a larger model is used

# Other useful options:
# --log_samples \
# --model_args pretrained=$model,load_in_8bit=True,peft=$lora_path,tokenizer=$tokenizer \
out_file_path="../../$save_path/out-$tasks-$JOBID.out"
output_path="../../$save_path/outputs_burns/$task.jsonl"
accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus \
    --mixed_precision=no --dynamo_backend=no \
    main.py \
    --model hf \
    --model_args pretrained=$model,load_in_8bit=True,peft=$lora_path \
    --tasks $task \
    --batch_size 8 \
    --output_path $output_path > $out_file_path


# ----------------------------------------
# Print the results
# ----------------------------------------
cd ../../
python mlmi-thesis/src/utils/get_harness_results_burns.py --output_path=$save_path


# ----------------------------------------
# Move the output file
# ----------------------------------------
cd mlmi-thesis/
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Log the duration
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
