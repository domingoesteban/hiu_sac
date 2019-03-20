#!/usr/bin/env bash


#log_dir='logs/centauro/sub-1/centauro-hiusac-p/centauro-hiusac-p--s-1010---2019_03_01_00_31_25'
#log_dir='logs/centauro/sub0/centauro-sac/centauro-sac--s-1010---2019_03_01_07_59_25'

#log_dir='logs/pusher/sub-1/pusher-hiusac-p/pusher-hiusac-p--s-5000---2019_03_05_14_15_14'
#log_dir='logs/pusher/sub-1/pusher-hiusac-p/pusher-hiusac-p--s-1010---2019_03_06_03_19_03'
#log_dir='logs/pusher/sub-1/pusher-hiusac-p/pusher-hiusac-p--s-1010---2019_03_06_02_44_50'
#log_dir='logs/pusher/sub-1/pusher-hiusac-p/pusher-hiusac-p--s-810---2019_03_05_21_38_34'
#log_dir='logs/pusher/sub-1/pusher-hiusac-p/pusher-hiusac-p--s-810---2019_03_05_21_39_00'

log_dir='logs/pusher/sub0/pusher-sac/pusher-sac--s-810---2019_03_06_00_11_30'
log_dir='logs/pusher/sub1/pusher-sac/pusher-sac--s-810---2019_03_06_02_45_49'


#horizon=500
horizon=300

# Policy subtask
#task=0
#task=1
task=-1

#env_task=0
#env_task=1
env_task=-1

#iterations=(0 25 50 75 100 125 150 175 200 225 250 275 -1)
#iterations=(0 25 50 75 100 125 150 175 -1)
iterations=(-1)


# Replace log_dir, pol-subtask, and env-subtask with script arguments
log_dir=${1-${log_dir}}
task=${2-${task}}
env_task=${3-${env_task}}


for itr_idx in ${!iterations[@]}; do
    itr=${iterations[itr_idx]}
    echo "Logging directory: ${log_dir}'"
    echo "********************************************************"

    eval_options="
    ${log_dir}
    --iteration ${itr}
    --horizon ${horizon}
    --env_task ${env_task}
    --task ${task}
    --option re
    "
    echo ${eval_options}

    # Run training script
    python eval.py ${eval_options}
#    echo eval.py ${eval_options}

done
