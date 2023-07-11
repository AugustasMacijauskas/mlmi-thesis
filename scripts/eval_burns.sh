#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=02:30:00
#SBATCH --mail-type=NONE


# ----------------------------------------
# 
# My own code starts below
# 
# ----------------------------------------


# ----------------------------------------
# Environment configuration
# ----------------------------------------
export ELK_DIR="/fsx/home-augustas/VINC-logs"
conda env list | grep "*"
python --version


# ----------------------------------------
# Configuring GPUs
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"

num_gpus=$(echo $cuda_devices | awk -F, '{print NF}')
echo "num_gpus: $num_gpus"

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
keyword="dolly-v2-3b_imdb"

cd ..
now=$(date "+%Y%m%d_%H%M%S")
save_path="logs_eval_burns/${keyword}_${now}_${JOBID}"
mkdir $save_path
cd $workdir

out_file_path="../$save_path/out.$JOBID"


# ----------------------------------------
# Model
# ----------------------------------------
# model="databricks/dolly-v2-3b"
model="gpt2-xl"


# ----------------------------------------
# Datasets
# ----------------------------------------
# all_datasets="ag_news,amazon_polarity,super_glue:boolq,super_glue:copa,dbpedia_14,imdb,piqa,glue:qnli,super_glue:rte"
all_datasets="ag_news,glue:qnli"


# ----------------------------------------
# Log the start time,
# cd in to the correct and
# start executing the commands
# ----------------------------------------
start=`date +%s`
cd ../elk


# ----------------------------------------
# Launch the commands
# ----------------------------------------

# Set the IFS to comma (,) to split the list
IFS=','

# Iterate over each dataset in the list
for dataset_name in $all_datasets; do
    echo "Processing dataset: $dataset_name"

    options="elicit $model $dataset_name --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 400000"
    CMD="elk $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD

    # Create a results file
    python src/utils/get_results.py --file_path=$out_file_path
done


# ----------------------------------------
# ag_news
# ----------------------------------------
options="elicit $model ag_news --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 7600"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# amazon_polarity
# ----------------------------------------
# options="elicit $model amazon_polarity --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 400000"
# CMD="elk $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD

# # Create a results file
# python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# dbpedia_14
# ----------------------------------------
# options="elicit $model dbpedia_14 --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 70000"
# CMD="elk $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD

# # Create a results file
# python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# glue:qnli
# ----------------------------------------
options="elicit $model glue:qnli --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 5463"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# imdb
# ----------------------------------------
options="elicit $model imdb --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 25000"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# piqa
# ----------------------------------------
options="elicit $model piqa --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 1838"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# super_glue:boolq
# ----------------------------------------
options="elicit $model super_glue:boolq --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 3270"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# super_glue:boolq
# ----------------------------------------
options="elicit $model super_glue:boolq --num_gpus=$num_gpus --min_gpu_mem=0 --disable_cache --num_examples 100 3270"
CMD="elk $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path


# ----------------------------------------
# Move the output file
# ----------------------------------------
cd ../mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Log the duration
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
