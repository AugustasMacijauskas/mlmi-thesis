#!/bin/bash

#SBATCH -A trlx
#SBATCH -p g40
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=12
#SBATCH -J augustas-thesis
#SBATCH --time=20:00:00
#SBATCH --mail-type=NONE

# My own code

# ----------------------------------------
# GPU
# ----------------------------------------
cuda_devices=$(echo $CUDA_VISIBLE_DEVICES)  # Store the value of CUDA_VISIBLE_DEVICES in a variable
echo "CUDA_VISIBLE_DEVICES: $cuda_devices"
nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1

# ----------------------------------------
# Environment
# ----------------------------------------
conda env list | grep "*"
python --version

# ----------------------------------------
# Output path
# ----------------------------------------
workdir="$SLURM_SUBMIT_DIR"
cd $workdir

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

now=$(date "+%Y%m%d_%H%M%S")
keyword="gpt2-xl_rlfhed_old"
save_path="logs_eval/${keyword}_${now}_${JOBID}"
# save_path=$1
echo "Save path: $save_path"
cd ..
mkdir $save_path
mkdir "$save_path/outputs"
cd $workdir
out_file_path="../$save_path/out.$JOBID"

# ----------------------------------------
# Application
# ----------------------------------------
application="python"

# ----------------------------------------
# Model
# ----------------------------------------
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230704_091318_26861/checkpoints/model_step_6"
model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_20230711_080057_31473/checkpoints/model_step_12"
# model="/fsx/home-augustas/ppo_logs/gpt2-xl_unifiedqa_3b_imdb_20230708_234722_29602/checkpoints/model_step_10"
# model="gpt2-xl"

# Log the start time and start executing the commands
start=`date +%s`
cd ../lm-evaluation-harness

# ----------------------------------------
# ARC
# ----------------------------------------
task="arc_challenge"
shots="25"
options="main.py \
    --model=hf-causal \
    --model_args=pretrained=$model \
    --tasks=$task \
    --num_fewshot=$shots \
    --batch_size=16 \
    --output_path=/fsx/home-augustas/$save_path/outputs/$task-$shots-shot.json \
    --device cuda \
"
CMD="$application $options > $out_file_path"
echo -e "\nExecuting command:\n==================\n$CMD\n"
eval $CMD

# ----------------------------------------
# TruthfulQA
# ----------------------------------------
# tasks="truthfulqa_mc"
# shots="0"
# options="main.py \
#     --model=hf-causal \
#     --model_args=pretrained=$model \
#     --tasks=$tasks \
#     --num_fewshot=$shots \
#     --batch_size=64 \
#     --output_path=/fsx/home-augustas/$save_path/outputs/$tasks-$shots-shot.json \
#     --device cuda \
# " # Can add device here: --device cuda \
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD

# ----------------------------------------
# HellaSwag
# ----------------------------------------
# task="hellaswag"
# shots="10"
# options="main.py \
#     --model=hf-causal \
#     --model_args=pretrained=$model \
#     --tasks=$task \
#     --num_fewshot=$shots \
#     --batch_size=16 \
#     --output_path=/fsx/home-augustas/$save_path/outputs/$task-$shots-shot.json \
# " # Can add device here: --device cuda
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD

# ----------------------------------------
# MMLU
# ----------------------------------------
# task="mmlu"
# # tasks="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions"
# tasks="hendrycksTest-*"

# shots="5"
# options="main.py \
#     --model=hf-causal \
#     --model_args=pretrained=$model \
#     --tasks=$tasks \
#     --num_fewshot=$shots \
#     --batch_size=16 \
#     --output_path=/fsx/home-augustas/$save_path/outputs/$task-$shots-shot.json \
#     --device cuda \
# "
# CMD="$application $options > $out_file_path"
# echo -e "\nExecuting command:\n==================\n$CMD\n"
# eval $CMD

# Move the output file
cd ../mlmi-thesis
echo -e "\nMoving file slurm-$JOBID.out to $save_path"
mv slurm-$JOBID.out ../$save_path

# Log the duration
end=`date +%s`
duration=$((end-start))
duration=`date -u -d @${duration} +"%T"`
echo -e "\nRuntime: $duration"
