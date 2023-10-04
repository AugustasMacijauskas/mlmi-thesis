#!/bin/bash

# Some guidelines for good default options are:
# 1 node, 8 GPUs, 12 CPUs per GPU, 1 hour
# To get email notifications use the --mail-type option

#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J <job-name>
#SBATCH --time=01:00:00


# ----------------------------------------
# 
# My own code starts below
# 
# ----------------------------------------


# ----------------------------------------
# Environment configuration
# ----------------------------------------
export ELK_DIR="../elk-probes"
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
keyword="UQA-3B-v2-test" # Edit to your liking

save_path="logs_elk/${keyword}_${now}_${JOBID}"
cd ..
mkdir $save_path
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
model="allenai/unifiedqa-v2-t5-3b-1363200"
echo "Model: $model"


# ----------------------------------------
# Dataset
# ----------------------------------------
dataset="AugustasM/qnli-vicuna-v1"
echo "Dataset: $dataset"


# ----------------------------------------
# Template
# ----------------------------------------
template_path="AugustasM/burns-datasets-VINC"
# template_path="AugustasM/burns-datasets-VINC/all"
echo -e "Template path: $template_path\n"


# ----------------------------------------
# Launch the command
# ----------------------------------------

# MAKE SURE TO EDIT THE OPTIONS BELOW TO YOUR LIKING

# Other available options:
# --net.seed=0 \
# --data.seed=0 \
# --net ccs \
# --disable_cache \
# --max_examples 10000 2000 \
# --supervised {none,single,inlp,cv} \
# --prompt_indices $template_number \
# --use_encoder_states \

options="elicit $model $dataset \
    --num_gpus=$num_gpus --min_gpu_mem=0 \
    --supervised single \
    --template_path=$template_path \
"

out_file_path="../$save_path/out.$JOBID"
CMD="elk $options > $out_file_path"

# Log the start time and execute the command
start=`date +%s`
cd ../elk
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD


# ----------------------------------------
# Move the output file - DO NOT EDIT
# ----------------------------------------
cd ../mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Create a results file - DO NOT EDIT
# ----------------------------------------
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# Log the duration - DO NOT EDIT
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
