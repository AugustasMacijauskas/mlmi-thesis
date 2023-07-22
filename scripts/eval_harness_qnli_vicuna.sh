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
# Configuring GPUs
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "num_gpus: $num_gpus"

export NUMEXPR_MAX_THREADS="$((num_gpus * 12))"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"


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
keyword="vicuna"
save_path="logs_eval/${keyword}_${now}_${JOBID}"

cd ..
mkdir $save_path
mkdir "$save_path/outputs_burns"
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
model="lmsys/vicuna-7b-v1.3"
echo "Model: $model"


# ----------------------------------------
# The task
# ----------------------------------------
task="qnli_vicuna"
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
out_file_path="../../$save_path/out-$tasks-$JOBID.out"
output_path="../../$save_path/outputs/$task.jsonl"
accelerate launch --multi_gpu --num_machines=1 --num_processes=$num_gpus \
    --mixed_precision=no --dynamo_backend=no \
    main.py \
    --model hf \
    --model_args pretrained=$model,load_in_8bit=True \
    --tasks $task \
    --batch_size 8 \
    --log_samples \
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
