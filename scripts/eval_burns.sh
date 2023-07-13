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
conda env list | grep "*"
python --version
squeue --me | grep $SLURM_JOB_ID


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
# keyword="dolly-v2-3b_imdb"
# keyword="gpt2-xl_test"
# keyword="gpt2-xl"
# keyword="gpt2-xl_rlhfed"
# keyword="gpt2-xl_imdb_rlhfed"

cd ..
now=$(date "+%Y%m%d_%H%M%S")
# save_path="logs_eval_burns/${keyword}_${now}_${JOBID}"
save_path="logs_eval_burns/gpt2-xl_rlhfed_20230712_143355_33041"
mkdir $save_path
cd $workdir


# ----------------------------------------
# Model
# ----------------------------------------
# model="databricks/dolly-v2-3b"
# model="gpt2-xl"
model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230704_091318_26861/checkpoints/model_step_6"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230711_080057_31473/checkpoints/model_step_12"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230708_234722_29602/checkpoints/model_step_10"
echo -e "model: $model\n"


# ----------------------------------------
# Datasets
# ----------------------------------------
# all_datasets="ag_news,amazon_polarity,super_glue:boolq,super_glue:copa,dbpedia_14,imdb,piqa,glue:qnli,super_glue:rte"
# all_datasets="super_glue:copa,super_glue:rte"
all_datasets="imdb,piqa,glue:qnli,super_glue:rte"
# all_datasets="imdb"
echo -e "all_datasets: $all_datasets\n"


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
    echo -e "\nProcessing dataset: $dataset_name"

    out_dir="/fsx/home-augustas/$save_path/$dataset_name"
    echo "Output directory: $out_dir"

    options="elicit $model $dataset_name \
        --num_gpus=$num_gpus --min_gpu_mem=0 \
        --layers=1 \
        --supervised=none \
        --disable_cache \
        --max_examples 5 10000 \
        --out_dir=$out_dir \
    "
    out_file_path="../$save_path/out-$dataset_name-$JOBID.out"
    CMD="elk $options > $out_file_path"
    echo -e "\nExecuting command:\n==================\n$CMD\n"
    eval $CMD

    # Create a results file
    python ../mlmi-thesis/src/utils/get_results_burns.py --file_path=$out_file_path --suffix=$dataset_name
done


# ----------------------------------------
# Average out the results
# ----------------------------------------
cd ..
python mlmi-thesis/src/utils/get_results_burns_averaged.py --output_path=$save_path


# ----------------------------------------
# Move the output file
# ----------------------------------------
cd mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path


# ----------------------------------------
# Log the duration
# ----------------------------------------
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
