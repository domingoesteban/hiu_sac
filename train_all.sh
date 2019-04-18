#!/usr/bin/env bash

# This bash script runs all the experiments conducted in the paper.
# WARNING: Running all the experiments can take several hours/days depending on
#          your CPU/GPU. However, by modifying the script variables you can run
#          whatever you are interested in.

# The first script argument is a GPU ID. E.g.
# $ bash training_script 0
# or if none is provided the cpu is used:
# $ bash training_script

# ################ #
# Script Variables #
# ################ #
# You can modify these variables

# Environment names
#env_names=('navigation2d' 'reacher' 'pusher' 'centauro')  # Array of environments
#env_names=('navigation2d')  # Uncomment for a single environment
env_names=('centauro')  # Uncomment for a single environment

# Algorithm names
# hiusac-p: HIU-2 in paper | hiusac: HIU-1 in paper | 'sac': single-task SAC
algo_names=('hiusac-p' 'hiusac' 'sac')  # Array of algorithms
#algo_names=('hiusac-p')  # Uncomment for a single algorithm

# Experiment seeds
#seeds=(610 710 810 910 1010 1110 1210 1310 1510 1610)  # Array of seeds
seeds=(610 710 810 910 1010)  # Array of seeds
#seeds=(810)  # Uncomment for a single seed

# Environment (sub) tasks
# -1: Main task | 0: First subtask | 1: Second subtask
subtasks=(-1 0 1)  # Array of subtasks
#subtasks=(-1)  # Uncomment for a single subtask

# Log also in stdout
log_stdout=false

# ##################################### #
# WARNING: DO NOT MODIFY THE FOLLOWING. #
# ##################################### #
# Variables used in the script
gpu_id=${1-cpu}  # Get first script argument. CPU or GPU option.
total_scripts=$((${#env_names[@]} * ${#algo_names[@]} * ${#subtasks[@]} * ${#seeds[@]}))
script_counter=0

echo "HIU-SAC paper script"
echo "Total environments: ${#env_names[@]}"
echo "Environment names: ${env_names[@]}"
echo "--"
echo "Total different algorithms: ${#algo_names[@]}"
echo "Algorithm names: ${algo_names[@]}"
echo "--"
echo "Total different subtasks: ${#subtasks[@]}"
echo "Algorithm names: ${subtasks[@]}"
echo "--"
echo "Total seeds: ${#seeds[@]}"
echo "Experiment seeds: ${seeds[@]}"
echo "--"
echo ""

for seed_idx in ${!seeds[@]}; do
for subtask_idx in ${!subtasks[@]}; do
for env_idx in ${!env_names[@]}; do
for algo_idx in ${!algo_names[@]}; do

    # Get variable values
    algo_name=${algo_names[algo_idx]}
    seed=${seeds[seed_idx]}
    subtask=${subtasks[subtask_idx]}
    env_name=${env_names[env_idx]}
    script_counter=$((script_counter+1))

    # Define the log directory
    log_dir_path='./training_logs/'${env_name}'/'  # Log directory path
    dir_prefix=${algo_name}  # Log directory prefix

    echo "********************************************************"
    echo "********************************************************"
    echo "Running 'train.py' script:${script_counter}/${total_scripts}"
    echo "Env: ${env_name} | Algo: ${algo_name} | Seed: ${seed} | Subtask: ${subtask}"

    expt_name='sub'${subtask}_${algo_name}_${seed}
    log_dir=${log_dir_path}'sub'${subtask}
    echo "Logging directory: ${log_dir}'"

    # Specify the training options
    training_options="
    --log_dir ${log_dir}
    --env_name ${env_name}
    --algo ${algo_name}
    --seed ${seed}
    --subtask ${subtask}
    "
    if [ ${gpu_id} != cpu ]; then
        training_options+=" --gpu ${gpu_id}"
    fi
    if [ ${log_stdout} = false ] ; then
        training_options+=" --no_log_stdout"
    fi
    echo "Training options: ${training_options}"
    echo "********************************************************"

    # Run the training script with the specified training options
    python train.py ${training_options}

    echo "********************************************************"
    echo ""

done
done
done
done
