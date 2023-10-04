#!/bin/bash

# Some guidelines for good default options are:
# 1 node, 8 GPUs, 12 CPUs per GPU, 10 hours
# To get email notifications use the --mail-type option

#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J <job-name>
#SBATCH --time=10:00:00


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
keyword="vicuna-7B-openllm"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_open_llm"
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
lora_path="/fsx/home-augustas/ppo_logs/vicuna-v1.5_UQA_3b_qnli_20230805_144210_53882/checkpoints/model_step_1_64"
# lora_path=""
echo "LoRA path: $lora_path"


# ----------------------------------------
# All tasks
# ----------------------------------------
open_llm_leaderboard_tasks="arc_challenge,hellaswag,truthfulqa_mc,mmlu"
open_llm_leaderboard_tasks="arc_challenge,truthfulqa_mc"
echo -e "Open LLM leaderboard tasks: $open_llm_leaderboard_tasks\n"


# ----------------------------------------
# Batch sizes - a lookup table
# ----------------------------------------
declare -A batch_sizes
batch_sizes["arc_challenge"]="16"
batch_sizes["hellaswag"]="16"


# ----------------------------------------
# Log the start time,
# cd in to the correct and
# start executing the commands
# ----------------------------------------
start=`date +%s`
cd ../lm_evaluation_harness_refactored/lm-evaluation-harness


# ----------------------------------------
# Open LLM Leaderboard tasks
# ----------------------------------------

# Set the IFS to comma (,) to split the list
IFS=','

declare -A num_few_shot_examples
num_few_shot_examples["arc_challenge"]="25"
num_few_shot_examples["hellaswag"]="10"

# Iterate over each task in the list
for task in $open_llm_leaderboard_tasks; do
    # Skip truthfulqa_mc
    if [[ "$task" == "truthfulqa_mc" ]]; then
        continue
    fi

    if [[ "$task" == "mmlu" ]]; then
        continue
    fi

    echo -e "\nEvaluating on task: $task"

    output_path="../../$save_path/outputs_open_llm/$task-$shots-shot.jsonl"
    echo "Output directory: $output_path"

    # Get the batch size for the task
    batch_size=${batch_sizes[$task]}

    # Get the number of few shot examples for the task
    shots=${num_few_shot_examples[$task]}
    echo "Number of few shot examples: $shots"

    # Other useful options:
    # --log_samples \
    # --model_args pretrained=$model,load_in_8bit=True,peft=$lora_path,tokenizer=$tokenizer \
    out_file_path="../../$save_path/out-$task-$JOBID.out"
    accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus \
        --mixed_precision=no --dynamo_backend=no \
        main.py \
        --model hf \
        --model_args pretrained=$model,load_in_8bit=True,peft=$lora_path \
        --tasks $task \
        --num_fewshot $shots \
        --batch_size $batch_size \
        --output_path $output_path > $out_file_path

done


# ----------------------------------------
# TruthfulQA
# ----------------------------------------

# Check if "truthfulqa_mc" is in the list
if [[ $open_llm_leaderboard_tasks == *"truthfulqa_mc"* ]]; then
    cd /fsx/home-augustas/lm-evaluation-harness
    
    task="truthfulqa_mc"
    tasks="truthfulqa_mc"
    shots="0"
    
    # --model_args pretrained=$model,tokenizer=$tokenizer \
    # Reduce batch size for larger models, this is for a 7B model
    out_file_path="../$save_path/out-$task-$JOBID.out"
    python main.py \
        --model hf-causal-experimental \
        --model_args pretrained=$model,peft=$lora_path \
        --tasks $tasks \
        --num_fewshot $shots \
        --batch_size 32 \
        --output_path /fsx/home-augustas/$save_path/outputs_open_llm/$tasks-$shots-shot.jsonl \
        --device cuda > $out_file_path
fi


# ----------------------------------------
# MMLU
# ----------------------------------------

# Check if "mmlu" is in the list
if [[ $open_llm_leaderboard_tasks == *"mmlu"* ]]; then
    cd /fsx/home-augustas/lm-evaluation-harness
    
    task="mmlu"
    tasks="hendrycksTest-*"
    shots="5"
    
    out_file_path="../$save_path/out-$task-$JOBID.out"
    python main.py \
        --model hf-causal-experimental \
        --model_args pretrained=$model,peft=$lora_path \
        --tasks $tasks \
        --num_fewshot $shots \
        --batch_size 4 \
        --output_path /fsx/home-augustas/$save_path/outputs_open_llm/$task-$shots-shot.json \
        --device cuda > $out_file_path
fi


# ----------------------------------------
# Average out the Open LLM results
# ----------------------------------------
if [[ -n "$open_llm_leaderboard_tasks" ]]; then
    cd /fsx/home-augustas/
    python mlmi-thesis/src/utils/get_harness_results.py --output_path=$save_path
fi

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
