#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=04:00:00
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
# keyword="gpt2-xl_rlhfed_imdb_openllm"
keyword="gpt2-xl_test_copa"
save_path="logs_eval_burns/${keyword}_${now}_${JOBID}"
cd ..
mkdir $save_path
mkdir "$save_path/outputs"
cd $workdir
out_file_path="../../$save_path/out.$JOBID"


# ----------------------------------------
# Model
# ----------------------------------------
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230704_091318_26861/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230711_080057_31473/checkpoints/model_step_12"
model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230708_234722_29602/checkpoints/model_step_10"
# model="gpt2-xl"
# model="databricks/dolly-v2-3b"
echo "Model: $model"


# ----------------------------------------
# All tasks
# ----------------------------------------
all_tasks="boolq_custom,copa_custom,qnli,rte"
echo -e "All tasks: $all_tasks\n"


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
application="accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus --mixed_precision=no --dynamo_backend=no"


# ----------------------------------------
# BoolQ
# ----------------------------------------
# task="boolq_custom"
# options="main.py \
#     --model=hf \
#     --model_args=pretrained=$model \
#     --tasks=$task \
#     --batch_size=32 \
#     --output_path=../../$save_path/outputs/$task.jsonl \
# "
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD


# ----------------------------------------
# QNLI
# ----------------------------------------
# task="qnli"
# options="main.py \
#     --model=hf \
#     --model_args=pretrained=$model \
#     --tasks=$task \
#     --batch_size=32 \
#     --output_path=../../$save_path/outputs/$task.jsonl \
# "
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD


# ----------------------------------------
# RTE
# ----------------------------------------
# task="rte"
# options="main.py \
#     --model=hf \
#     --model_args=pretrained=$model \
#     --tasks=$task \
#     --batch_size=32 \
#     --output_path=../../$save_path/outputs/$task.jsonl \
# "
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD


# ----------------------------------------
# Copa
# ----------------------------------------
task="copa"
options="main.py \
    --model=hf-causal \
    --model_args=pretrained=$model \
    --tasks=$task \
    --batch_size=32 \
    --output_path=../$save_path/outputs/$task.jsonl \
    --device cuda \
"
cd ../../lm-evaluation-harness
application="python"
out_file_path="../$save_path/out.$JOBID"
CMD="$application $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD


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
