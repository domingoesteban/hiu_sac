#!/usr/bin/env bash

# #### #
# VARS #
# #### #
gpu_id=${1-cpu}

#env_names=('navigation2d')
#env_names=('reacher')
env_names=('pusher')
#env_names=('centauro')
#env_names=('centauro' 'reacher')
total_envs=${#env_names[@]}

algo_name='sac'
#algo_name='hiusac'  # HIU-1
#algo_name='hiusac-p'  # HIU-2
#algo_name='hiusac-m'

dir_prefix=${algo_name}
#dir_prefix=${algo_name}


#seeds=(610 710 810 910 1010)
#seeds=(710)
#seeds=(810 1010)
#seeds=(5000)
seeds=(810 1010)
#seeds=(610)
total_seeds=${#seeds[@]}

subtasks=(-1 0 1)
#subtasks=(0 1)
#subtasks=(1)
#subtasks=(-1)
#subtasks=("${@:-${default_subtasks[@]}}")
total_subtasks=${#subtasks[@]}


total_scripts=$(($total_seeds * $total_subtasks))

echo "Robolearn DRL script"
echo "Total seeds: ${#seeds[@]}"
echo "Experiment seeds: ${seeds[@]}"
echo ""

for env_idx in ${!env_names[@]}; do
for seed_idx in ${!seeds[@]}; do
for subtask_idx in ${!subtasks[@]}; do
    seed=${seeds[seed_idx]}
    subtask=${subtasks[subtask_idx]}
    env_name=${env_names[env_idx]}

    python_script=${env_name}'_'${algo_name}
    log_dir_path='./logs/'${env_name}'/'

#    script_index=$((index+init_index))
    script_index=$(((seed_idx)*total_subtasks + subtask_idx))
    echo "********************************************************"
    echo "Running 'train.py' script:$((script_index+1))/${total_scripts}
    | Seed: ${seed} | Subtask: ${subtask}"

    expt_name='sub'${subtask}_${algo_name}_${seed}
    log_dir=${log_dir_path}'sub'${subtask}
    echo "Logging directory: ${log_dir}'"
    echo "********************************************************"

    training_options="
    --log_dir ${log_dir}
    --seed ${seed}
    --env ${env_name}
    --algo ${algo_name}
    --task ${subtask}
    "
    if [ "$gpu_id" != cpu ]; then
        training_options+=" --gpu ${gpu_id}"
    fi
    echo ${training_options}

    # Run training script
    python train.py ${training_options}

done
done
done
