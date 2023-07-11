#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE

# My own code
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')

export NUMEXPR_MAX_THREADS="$((num_gpus * 12))"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"

nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1

conda env list | grep "*"
python --version

workdir="$SLURM_SUBMIT_DIR"
cd $workdir

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

now=$(date "+%Y%m%d_%H%M%S")
# version="v2"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_imdb_${version}_first"
keyword="gpt2-xl_rlhfed_imdb_openllm"
cd ..
save_path="logs_eval/${keyword}_${now}_${JOBID}"
mkdir $save_path
mkdir "$save_path/outputs"
cd $workdir
out_file_path="../../$save_path/out.$JOBID"

# Application and its run options:
application="accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus --mixed_precision=no --dynamo_backend=no"

# ----------------------------------------
# Model
# ----------------------------------------
model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230708_234722_29602/checkpoints/model_step_10"
# model="/fsx/home-augustas/ppo_runs/gpt2-xl_unifiedqa_3b_20230704_091318_26861/model_step_6"
# model="gpt2-xl"

# Log the start time and start executing the commands
start=`date +%s`
cd ../lm_evaluation_harness_refactored/lm-evaluation-harness

# ----------------------------------------
# ARC
# ----------------------------------------
task="arc_challenge"
shots="25"
options="main.py \
    --model=hf \
    --model_args=pretrained=$model \
    --tasks=$task \
    --num_fewshot=$shots \
    --batch_size=16 \
    --output_path=../../$save_path/outputs/$task-$shots-shot.jsonl \
" # Can add device here: --device cuda:0
CMD="$application $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# ----------------------------------------
# HellaSwag
# ----------------------------------------
task="hellaswag"
shots="10"
options="main.py \
    --model=hf \
    --model_args=pretrained=$model \
    --tasks=$task \
    --num_fewshot=$shots \
    --batch_size=16 \
    --output_path=../../$save_path/outputs/$task-$shots-shot.jsonl \
" # Can add device here: --device cuda
CMD="$application $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Move the output file
cd ../../mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path

# Log the duration
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
