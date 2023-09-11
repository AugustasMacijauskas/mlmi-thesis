#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40x
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
keyword="vicuna-13B-openllm"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_open_llm"
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
# model="lmsys/vicuna-7b-v1.3"
# model="lmsys/vicuna-7b-v1.5"
# model="lmsys/vicuna-13b-v1.3"
model="lmsys/vicuna-13b-v1.5"
# model="AugustasM/vicuna_rlhfed_v1"
# model="AugustasM/vicuna-v1.5-rl-qnli-v1"
echo "Model: $model"


# ----------------------------------------
# Tokenizer
# ----------------------------------------
# tokenizer="lmsys/vicuna-7b-v1.3"
# tokenizer="lmsys/vicuna-7b-v1.5"
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

# lora_path="/fsx/home-augustas/ppo_logs/vicuna-13B-UQA-3B_20230811_011047_60250/checkpoints/model_step_1_32"


# ---------------------------------------- Full LoRA ----------------------------------------
# 7B-v1.3
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230810_230806_59513/checkpoints/model_step_1_16"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230810_235545_59747/checkpoints/model_step_1_25"

# 7B-v1.5
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230810_230816_59514/checkpoints/model_step_1_16"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230811_000657_59889/checkpoints/model_step_1_25"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230810_230816_59514/checkpoints/model_step_1_32"
# lora_path="/fsx/home-augustas/ppo_logs/vicuna-full-LoRA-UQA-3B_20230810_230816_59514/checkpoints/model_step_1_48"


# lora_path=""
echo "LoRA path: $lora_path"


# ----------------------------------------
# All tasks
# ----------------------------------------
# open_llm_leaderboard_tasks="arc_challenge,hellaswag,truthfulqa_mc,mmlu"
# open_llm_leaderboard_tasks="arc_challenge,hellaswag,truthfulqa_mc"
# open_llm_leaderboard_tasks="arc_challenge,hellaswag"
# open_llm_leaderboard_tasks="mmlu"
open_llm_leaderboard_tasks="truthfulqa_mc"
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

    # --model_args pretrained=$model,load_in_8bit=True,peft=$lora_path \
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
    
    # --model_args pretrained=$model,peft=$lora_path \
        # --model_args pretrained=$model,tokenizer=$tokenizer \
    out_file_path="../$save_path/out-$task-$JOBID.out"
    python main.py \
        --model hf-causal-experimental \
        --model_args pretrained=$model,peft=$lora_path \
        --tasks $tasks \
        --num_fewshot $shots \
        --batch_size 16 \
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
        # --model_args pretrained=$model,tokenizer=$tokenizer,use_accelerate=True,load_in_8bit=True \
        # --model_args pretrained=$model,peft=$lora_path,use_accelerate=True,load_in_8bit=True \
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
