#!/bin/bash

save_path="logs_eval/gpt2-xl_rlhfed_imdb_openllm_20230710_201339_31000"
sbatch scripts/eval_harness.sh "$save_path"

squeue --me
