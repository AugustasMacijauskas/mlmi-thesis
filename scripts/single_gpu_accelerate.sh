#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=02:30:00
#SBATCH --mail-type=NONE

#! My own code
conda env list | grep "*"
python --version
nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1

workdir="$SLURM_SUBMIT_DIR"
cd $workdir

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

now=$(date "+%Y%m%d_%H%M%S")
keyword="gpt2-xl_unifiedqa_3b"

cd ..
save_path_stem="${keyword}_${now}_${JOBID}"
save_path="ppo_logs/$save_path_stem"
mkdir $save_path
cd $workdir

# Application and its run options:
application="accelerate"

# model="databricks/dolly-v2-3b"
model="gpt2-xl"
reward_model_output_path="/fsx/home-augustas/logs/unifiedqa-v2-t5-3b-1363200_custom_data_v4_all_20230629_120158_21789"
dataset="AugustasM/burns-datasets-VINC-ppo-training-v4"

options="launch --num_machines=1 --num_processes=1 --mixed_precision=no --dynamo_backend=no src/mvp_0/ppo_training.py \
    --model_name=$model \
    --tokenizer_name=$model \
    --reward_model_output_path=$reward_model_output_path \
    --dataset_name=$dataset \
    --remove_unused_columns=False \
    --log_with=tensorboard \
    --logging_dir=/fsx/home-augustas/ppo_tensorboard_logs/$save_path_stem/ \
    --learning_rate=1.4e-5 \
    --batch_size=64 \
    --mini_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --steps=128 \
    --ppo_epochs=4 \
    --early_stopping=True \
    --reward_baseline=0.0 \
    --target_kl=0.1 \
    --init_kl_coef=0.2 \
    --adap_kl_ctrl=True \
    --seed=0 \
    --save_freq=1000 \
    --output_dir=/fsx/home-augustas/ppo_runs/$save_path_stem/model_ \
"

out_file_path="../$save_path/out.$JOBID"
CMD="$application $options > $out_file_path"

# Log the start time and execute the command
start=`date +%s`
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Move the output file
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path

# Log the duration
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
