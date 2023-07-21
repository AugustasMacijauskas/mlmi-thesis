#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=20:00:00
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
nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1


# ----------------------------------------
# Configuring GPUs
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

# Use awk to get the index of the last GPU
last_gpu_index=$(echo $cuda_devices | awk -F, '{print $NF}')
echo "Last GPU index: $last_gpu_index"

# Use awk to get the number of available devices
total_num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
ppo_gpus=$((total_num_gpus - 1)) # number of GPUs for PPO is total_num_gpus - 1
echo "Total number of GPUs: $total_num_gpus"
echo "Number of GPUs for PPO: $ppo_gpus"


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
model="gpt2-xl"
# keyword="${model}_unifiedqa_3b_custom_data_v4"
keyword="${model}_unifiedqa_3b_imdb"

cd ..
save_path_stem="${keyword}_${now}_${JOBID}"
save_path="ppo_logs_trlx/$save_path_stem"
mkdir $save_path
cd $workdir


# ----------------------------------------
# Launch the command
# ----------------------------------------
application="accelerate"

    # --config_file /fsx/home-augustas/mlmi-thesis/scripts/configs/zero2-bf16.yaml \
# options="launch --num_machines=1 --num_processes=1 \
options="launch --multi_gpu --num_machines=1 --num_processes=$ppo_gpus \
    --config_file scripts/configs/zero2-bf16.yaml \
    --mixed_precision=no --dynamo_backend=no \
    src/ppo/ppo_training_advanced.py \
"

out_file_path="../$save_path/out.$JOBID"
CMD="$application $options > $out_file_path"


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
