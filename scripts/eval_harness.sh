#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=02:30:00
#SBATCH --mail-type=NONE

# My own code
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"
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
# keyword="unifiedqa_t5_3b_rte_ccs"
# keyword="unifiedqa_t5_3b_custom_data"
# keyword="unifiedqa-v2-t5-large-1363200_custom_data_first_encoder"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_all_more_data"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_v3_first_more_data_ccs"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_v3_all_more_data_ccs"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_v4_all"
# version="v2"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_imdb_${version}_first"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_imdb_${version}_all"
# keyword="unifiedqa-v2-t5-3b-1363200_custom_data_imdb_${version}_all_more_data"
# keyword="unifiedqa-v2-t5-11b-1363200_custom_data_all_more_data"
# keyword="unifiedqa-v2-t5-11b-1363200_custom_data_v2_all_ccs"
# keyword="unifiedqa-v2-t5-11b-1363200_custom_data_v3_all_ccs"
# keyword="unifiedqa-v2-t5-11b-1363200_custom_data_v4_all"
# keyword="unifiedqa-v2-t5-11b-1363200_individual_imdb_first"
# keyword="unifiedqa-v2-t5-11b-1363200_custom_data_imdb_${version}_first"
# keyword="deberta_v3_large_ag_news"
# keyword="deberta_v3_large_custom_data_first"
# keyword="deberta-v2-xxlarge_test"
# keyword="deberta_v3_large_custom_data_ag_news"
# keyword="gpt-j-6b_truthful_qc_mc"
# keyword="gpt-j-6b_custom_data"
# keyword="dolly-v2-3b_custom_data_v2_all"
# keyword="dolly-v2-3b_custom_data_v3_all"
# keyword="dolly-v2-3b_truthful_qc_mc"
# keyword="dolly-v2-3b_individual_imdb_first"
# keyword="gpt2_custom_data_v4_all"
# keyword="gpt2-xl_individual_imdb_first"
# keyword="gpt2-xl_imdb"
keyword="gpt2-xl_rlhfed_imdb"
# keyword="gpt2-xl_rlhfed_custom_data_v4_all"
cd ..
save_path="logs/${keyword}_${now}_${JOBID}"
mkdir $save_path
cd $workdir

# Application and its run options:
application="elk"

# ----------------------------------------
# Model
# ----------------------------------------
# model="gpt2"
# model="gpt2-xl"
model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230708_234722_29602/checkpoints/model_step_10"
# model="EleutherAI/gpt-j-6b"
# model="databricks/dolly-v2-3b"
# model="allenai/unifiedqa-t5-3b"
# model="allenai/unifiedqa-v2-t5-large-1363200"
# model="allenai/unifiedqa-v2-t5-3b-1363200"
# model="allenai/unifiedqa-v2-t5-11b-1363200"
# model="microsoft/deberta-v3-large"
# model="microsoft/deberta-v2-xxlarge"

dataset="imdb"
# dataset="super_glue:rte"
# dataset="AugustasM/burns-datasets-VINC"
# dataset="AugustasM/burns-datasets-VINC-v2"
# dataset="AugustasM/burns-datasets-VINC-v3"
# dataset="AugustasM/burns-datasets-VINC-v4"
# dataset="AugustasM/burns-datasets-VINC-$version"
# dataset="AugustasM/burns-datasets-VINC-imdb-$version"
# dataset="EleutherAI/truthful_qa_mc"
# dataset="truthful_qa_mc"
# dataset="AugustasM/burns-datasets-VINC-ag_news"
# dataset="AugustasM/burns-datasets-VINC-individual-imdb"

# template_path="AugustasM/burns-datasets-VINC/first"
# template_path="AugustasM/burns-datasets-VINC/all"
# template_path="reaganjlee/truthful_qa_mc"

# reporter_path="databricks/dolly-v2-3b/AugustasM/burns-datasets-VINC/reverent-yalow"

# options=$1
# options="elicit $model $dataset --num_gpus=1 --max_examples 100 100"
# options="elicit $model $dataset --num_gpus=1 --use_encoder_states"
# options="elicit $model $dataset --num_gpus=1 --net=ccs"
options="elicit $model $dataset --num_gpus=1"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --num_shots=1"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --disable_cache"
# options="elicit $model $dataset --num_gpus=1 --disable_cache"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --use_encoder_states"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --use_encoder_states --net=ccs"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --net=ccs"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --max_examples 1000 2000"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path --max_examples 10000 2000"
# options="elicit $model $dataset --num_gpus=1 --template_path=$template_path"
# options="eval $reporter_path $model $dataset --num_gpus=1 --template_path=$template_path"

out_file_path="../$save_path/out.$JOBID"
CMD="$application $options > $out_file_path"

# Log the start time and execute the command
start=`date +%s`
cd ../elk
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# Move the output file
cd ../mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path

# Create a results file
python src/utils/get_results.py --file_path=$out_file_path

# Log the duration
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"