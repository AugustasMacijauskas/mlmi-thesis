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
keyword="gpt2-xl_test_qnli"
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
# all_tasks="ag_news,amazon_polarity,boolq_custom,copa_custom,dbpedia_14,imdb,qnli,rte_custom"
# all_tasks="ag_news,boolq_custom,copa_custom,imdb,qnli,rte_custom"
all_tasks="boolq_custom,copa_custom,rte_custom"
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

# Set the IFS to comma (,) to split the list
IFS=','

# Iterate over each task in the list
for task in $all_tasks; do
    echo -e "\nEvaluating on task: $task"

    output_path="../../$save_path/outputs/$task.jsonl"
    echo "Output directory: $output_path"

    options="main.py \
        --model=hf \
        --model_args=pretrained=$model \
        --tasks=$task \
        --batch_size=32 \
        --output_path=$output_path \
        --no_sample_logging \
    "
    out_file_path="../../$save_path/out-$task-$JOBID.out"
    CMD="$application $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD
done


# ----------------------------------------
# ag_news
# ----------------------------------------
# task="ag_news"
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
# amazon_polarity
# ----------------------------------------
task="amazon_polarity"
options="main.py \
    --model=hf \
    --model_args=pretrained=$model \
    --tasks=$task \
    --batch_size=64 \
    --output_path=../../$save_path/outputs/$task.jsonl \
    --no_sample_logging \
"
CMD="$application $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD


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
# dbpedia_14
# ----------------------------------------
# task="dbpedia_14"
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
# imdb
# ----------------------------------------
# task="imdb"
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
#     --no_sample_logging \
# "
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD


# ----------------------------------------
# RTE
# ----------------------------------------
# task="rte_custom"
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
# task="copa"
# options="main.py \
#     --model=hf-causal \
#     --model_args=pretrained=$model \
#     --tasks=$task \
#     --batch_size=32 \
#     --output_path=../$save_path/outputs/$task.jsonl \
#     --device cuda \
# "
# cd ../../lm-evaluation-harness
# application="python"
# out_file_path="../$save_path/out.$JOBID"
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD


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
