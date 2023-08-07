#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=03:00:00
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


# ----------------------------------------
# Configuring GPUs
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "num_gpus: $num_gpus"

export NUMEXPR_MAX_THREADS="$((num_gpus * 12))"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"

nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1


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
# Save path
# ----------------------------------------
now=$(date "+%Y%m%d_%H%M%S")
keyword="vicuna-qnli-old-fixed"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_burns"
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
# model="lmsys/vicuna-7b-v1.3"
model="lmsys/vicuna-7b-v1.5"
# model="AugustasM/vicuna-v1.5-rl-qnli-v1"
echo "Model: $model"


# ----------------------------------------
# Tokenizer
# ----------------------------------------
# tokenizer="lmsys/vicuna-7b-v1.3"
tokenizer="lmsys/vicuna-7b-v1.5"
echo "Tokenizer: $tokenizer"


# ----------------------------------------
# LoRA weight path
# ----------------------------------------
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230802_185452_51334/checkpoints/model_step_1_8"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230803_093735_52310/checkpoints/model_step_1_12"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230802_142639_51052/checkpoints/model_step_1_6"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230803_122231_52395/checkpoints/model_step_1_40"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230803_144559_52437/checkpoints/model_step_1_16"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna_UQA_3b_qnli_20230803_144559_52437/checkpoints/model_step_1_32"

# lora_path="/fsx/home-augustas/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_16"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_32"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_48"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_64"
# lora_path=""
echo "LoRA path: $lora_path"


# ----------------------------------------
# The task
# ----------------------------------------
# task="qnli_vicuna"
task="qnli_custom_2"
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
    --batch_size 32 \
    --output_path $output_path > $out_file_path


# ----------------------------------------
# Print the results
# ----------------------------------------
cd /fsx/home-augustas/
python mlmi-thesis/src/utils/get_harness_results_burns.py --output_path=$save_path


# ----------------------------------------
# Move the output file
# ----------------------------------------
cd /fsx/home-augustas/mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Log the duration
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
