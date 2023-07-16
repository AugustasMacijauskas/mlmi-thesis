#!/bin/bash

The line below is a comment


# Declare an array of string with type
launch_argument_file="full_phones_2"
readarray -t options_array < "launch_arguments/$launch_argument_file.txt"

now=$(date "+%Y%m%d_%H%M%S")
common_save_path="${launch_argument_file}_$now"
export "common_save_path=${common_save_path}"

export "keyword=${common_save_path}"

# Iterate the string array using for loop
for options in "${options_array[@]}"; do
    export "options=${options}"
    echo $options
    sbatch launch_files/slurm_mlmi2_gpu_main
done

squeue -me
