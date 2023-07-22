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
# keyword="gpt2-xl_test_updates"
# keyword="gpt2-xl"
# keyword="gpt2-xl_rlhfed"
# keyword="gpt2-xl_rlhfed_short"
# keyword="gpt2-xl_rlhfed_long"
# keyword="gpt2-xl_imdb_rlhfed_supervised"
keyword="vicuna"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_burns"
mkdir "$save_path/outputs_open_llm"
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
# model="gpt2-xl"
# model="lmsys/vicuna-7b-v1.3"

# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_custom_data_v4_20230703_211833_26655/checkpoints/model_step_3"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_custom_data_v4_20230704_091318_26861/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_custom_data_v4_20230704_164952_27036/checkpoints/model_step_3"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_custom_data_v4_20230711_080057_31473/checkpoints/model_step_12"

# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230707_174421_29011/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230707_234807_29238/checkpoints/model_step_5"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230708_234722_29602/checkpoints/model_step_10"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230710_073009_30249/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230710_073009_30249/checkpoints/model_step_12"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230710_111607_30508/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230710_111607_30508/checkpoints/model_step_12"

# ----------- Supervised models -----------
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_supervised_20230718_003533_37165/checkpoints/model_step_4"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_supervised_20230718_003533_37165/checkpoints/model_step_8"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_supervised_20230718_003533_37165/checkpoints/model_step_12"
echo "Model: $model"


# ----------------------------------------
# All tasks
# ----------------------------------------
# burns_tasks="ag_news_binarized,boolq,copa,imdb,qnli_custom,rte_custom,dbpedia_14_binarized,amazon_polarity"
# burns_tasks="ag_news_binarized,dbpedia_14_binarized"
# burns_tasks="boolq,imdb,dbpedia_14_binarized"
# burns_tasks="imdb_ps3,imdb_ps4,imdb_burns_1,imdb_burns_2"
burns_tasks=""
echo "Burns tasks: $burns_tasks"

# open_llm_leaderboard_tasks="arc_challenge,hellaswag,truthfulqa_mc"
open_llm_leaderboard_tasks=""
echo -e "Open LLM leaderboard tasks: $open_llm_leaderboard_tasks\n"


# ----------------------------------------
# Batch sizes - a lookup table
# ----------------------------------------
declare -A batch_sizes
batch_sizes["amazon_polarity"]="64"
batch_sizes["ag_news_binarized"]="32"
batch_sizes["boolq"]="16"
batch_sizes["copa"]="32"
batch_sizes["dbpedia_14"]="32"
batch_sizes["dbpedia_14_binarized"]="16"
batch_sizes["imdb"]="16"
batch_sizes["imdb_ps3"]="16"
batch_sizes["imdb_ps4"]="16"
batch_sizes["imdb_burns_1"]="16"
batch_sizes["imdb_burns_2"]="16"
batch_sizes["qnli_custom"]="32"
batch_sizes["rte_custom"]="16"
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
# Launch the commands
# ----------------------------------------
application="accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus --mixed_precision=no --dynamo_backend=no"


# ----------------------------------------
# Burns tasks
# ----------------------------------------

# Set the IFS to comma (,) to split the list
IFS=','

# Iterate over each task in the list
for task in $burns_tasks; do
    echo -e "\nEvaluating on task: $task"

    output_path="../../$save_path/outputs_burns/$task.jsonl"
    echo "Output directory: $output_path"

    # Get the batch size for the task
    batch_size=${batch_sizes[$task]}

    options="main.py \
        --model hf \
        --model_args pretrained=${model} \
        --tasks $task \
        --batch_size $batch_size \
        --output_path $output_path \
    "
    out_file_path="../../$save_path/out-$task-$JOBID.out"
    CMD="$application $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD
done


# ----------------------------------------
# Average out the Burns results
# ----------------------------------------
if [[ -n "$burns_tasks" ]]; then
    cd /fsx/home-augustas/
    python mlmi-thesis/src/utils/get_harness_results_burns.py --output_path=$save_path
fi

# ----------------------------------------
# Open LLM Leaderboard tasks
# ----------------------------------------
cd /fsx/home-augustas/lm_evaluation_harness_refactored/lm-evaluation-harness

declare -A num_few_shot_examples
num_few_shot_examples["arc_challenge"]="25"
num_few_shot_examples["hellaswag"]="10"

# Iterate over each task in the list
for task in $open_llm_leaderboard_tasks; do
    # Skip truthfulqa_mc
    if [[ "$task" == "truthfulqa_mc" ]]; then
        continue
    fi

    echo -e "\nEvaluating on task: $task"

    output_path="../../$save_path/outputs_open_llm/$task-$shots-shot.jsonl"
    echo "Output directory: $output_path"

    # Get the batch size for the task
    batch_size=${batch_sizes[$task]}

    # Get the number of few shot examples for the task
    shots=${num_few_shot_examples[$task]}

    options="main.py \
        --model=hf \
        --model_args=pretrained=$model \
        --tasks=$task \
        --num_fewshot=$shots \
        --batch_size=$batch_size \
        --output_path=$output_path \
    "
    out_file_path="../../$save_path/out-$task-$JOBID.out"
    CMD="$application $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD
done


# ----------------------------------------
# TruthfulQA
# ----------------------------------------

# Check if "truthfulqa_mc" is in the list
if [[ $open_llm_leaderboard_tasks == *"truthfulqa_mc"* ]]; then
    tasks="truthfulqa_mc"
    shots="0"
    options="main.py \
        --model=hf-causal \
        --model_args=pretrained=$model \
        --tasks=$tasks \
        --num_fewshot=$shots \
        --batch_size=32 \
        --output_path=/fsx/home-augustas/$save_path/outputs_open_llm/$tasks-$shots-shot.jsonl \
        --device cuda \
    "
    cd /fsx/home-augustas/lm-evaluation-harness
    application="python"
    out_file_path="../$save_path/out-$task-$JOBID.out"
    CMD="$application $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD
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
