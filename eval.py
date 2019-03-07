import os.path as osp
import numpy as np
import torch
import argparse
import json


from envs import get_normalized_env
from utils import rollout
import plots


VIDEOS_DIR = 'videos'


# Numpy print options
np.set_printoptions(precision=3, suppress=True)

# Script parameters
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('log_dir', type=str,
                    help='Log directory')
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--env_task', '-e', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--horizon', '-n', type=int, default=None,
                    help='Rollout horizon [default: 100]')
parser.add_argument('--iteration', '-i', type=int, default=-1,
                    help='Model iteration [default: last]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--option', '-o', type=str, default=None,
                    help='Script option [default: None]')

n_rollouts = 0


def plot_progress(progress_file, algo_name='hiusac'):
    if algo_name in ['hiusac', 'hiusac-p']:
        num_intentions = 2
    else:
        num_intentions = None
    plots.plot_intentions_eval_returns(
        progress_file,
        num_intentions=num_intentions,
    )
    plots.plot_intentions_info(
        progress_file,
        num_intentions=num_intentions,
    )


def eval_policy(env, policy, max_horizon=50, task=None, stochastic=False,
                q_fcn=None):
    rollout_info = rollout(
        env,
        policy,
        max_horizon=max_horizon,
        fixed_horizon=False,
        device='cpu',
        render=True,
        intention=task,
        deterministic=not stochastic,
        return_info=True,
        q_fcn=q_fcn,
    )
    if task is None:
        rollout_return = sum(rollout_info['reward'])
    else:
        rollout_return = sum([info[task]
                              for info in rollout_info['reward_vector']])
    print("The rollout return is: %f" % rollout_return)


def record_policy(env, policy, max_horizon=50, task=None, stochastic=False,
                  q_fcn=None, video_name='rollout_video', video_dir=None):
    if video_dir is None:
        video_dir = VIDEOS_DIR

    video_name = osp.join(
        video_dir,
        video_name
    )
    if not video_name.endswith('.mp4'):
        video_name += '.mp4'

    rollout(
        env, policy,
        max_horizon=max_horizon,
        fixed_horizon=False,
        device='cpu',
        render=True,
        intention=task, deterministic=not stochastic,
        return_info=False,
        q_fcn=q_fcn,
        record_video_name=video_name,
    )


def plot_value_fcn(qf, policy, obs, action_lows, action_highs, actions_dims=(0, 1)):
    plots.plot_q_values(
        qf,
        action_lower=action_lows,
        action_higher=action_highs,
        obs=obs,
        policy=policy,
        action_dims=actions_dims,
        delta=0.05,
        device='cpu'
    )


if __name__ == '__main__':
    # Parse and print out parameters
    args = parser.parse_args()

    # Plot training data
    log_dir = args.log_dir
    progress_file = osp.join(log_dir, 'progress.csv')

    # Get environment from log directory
    with open(osp.join(log_dir, 'variant.json')) as json_data:
        log_data = json.load(json_data)
        env_name = log_data['env_name']
        env_params = log_data['env_params']
        algo_params = log_data['algo_params']
        seed = algo_params['seed']
        horizon = algo_params['max_horizon']
    env, env_params = get_normalized_env(
        env_name, args.task, args.seed, render=True, new_env_params=env_params
    )

    # Get models from file
    itr_dir = 'itr_%03d' % args.iteration if args.iteration > -1 else 'last_itr'
    models_dir = osp.join(log_dir, 'models', itr_dir)
    policy_file = osp.join(models_dir, 'policy.pt')
    qf_file = osp.join(models_dir, 'qf1.pt')
    policy = torch.load(policy_file, map_location=lambda storage, loc: storage)
    qf = torch.load(qf_file, map_location=lambda storage, loc: storage)

    options_dict = dict(
        p="plot progress",
        v="plot qval",
        e="evaluate",
        ie="evaluate infinite times!",
        r="record interaction",
        re="record and exit",
        n="navigation2d",
        t="change policy task",
        i="change iteration",
        et="change env task",
        h="change evaluation horizon",
        q="exit this script",
    )

    options_txt = "Select an option: \n"
    for key, val in options_dict.items():
        options_txt += "\t%s: %s\n" % (key, val)
    options_txt += "Option: "

    infinite_loop = False
    while True:
        if args.option is None:
            args.option = input(options_txt)

        if args.option.lower() == 'q':
            print("Closing the script. Bye!")
            break

        elif args.option.lower() == 'p':
            plot_progress(progress_file, log_data['algo_name'])

        elif args.option.lower() == 'v':
            obs = np.zeros(env.obs_dim)
            # TODO: Make this for all envs
            obs[0] = -6
            obs[1] = -6
            plot_value_fcn(qf, policy, obs, env.action_space.low, env.action_space.high)

        elif args.option.lower() == 'e':
            if args.horizon is not None:
                horizon = args.horizon
            eval_policy(env, policy,
                        max_horizon=horizon,
                        task=args.task,
                        stochastic=args.stochastic,
                        q_fcn=qf,
                        )

        elif args.option.lower() == 't':
            new_task = input("Specify task id (-1 for None). Task id: ")
            new_task = int(new_task)
            if new_task not in list(range(-1, policy.num_intentions)):
                print("Wrong option '%s'!" % new_task)
            args.task = None if new_task == -1 else new_task
            print("New task is %d" % new_task)

        elif args.option.lower() == 'et':
            new_task = input("Specify env_task id (-1 for None). Task id: ")
            try:
                new_task = int(new_task)
            except ValueError:
                print("Wrong option '%s'!. "
                      "It is not possible to convert it to int" % new_task)
            if new_task not in list(range(-1, env.n_subgoals)):
                print("Wrong option '%s'!" % new_task)
            else:
                new_task = None if new_task == -1 else new_task
                env.set_subtask(new_task)

        elif args.option.lower() == 'h':
            new_horizon = input("Specify new horizon: ")
            new_horizon = int(new_horizon)
            if not new_horizon > 1:
                print("Wrong horizon '%d'!" % new_horizon)
            else:
                args.horizon = new_horizon
                print("New horizon is %d" % new_horizon)

        elif args.option.lower() in ['r', 're']:
            max_iter = 300
            max_rollouts = 10
            # range_list = list(range(0, max_iter, 25)) + [None]
            range_list = [args.iteration]

            env_subtask = None if args.env_task == -1 else args.env_task
            env.set_subtask(env_subtask)

            for rr in range(max_rollouts):
                if args.horizon is not None:
                    horizon = args.horizon

                if env_subtask is None:
                    env_subtask = -1

                if args.task is None:
                    subtask = -1
                else:
                    subtask = args.task

                video_name = (
                        itr_dir +
                        ('_s%03d' % seed) +
                        ('_task%01d' % subtask) +
                        ('_envtask%01d' % env_subtask) +
                        ('_rollout%02d' % rr)
                )
                video_name = osp.join(
                    env_name,
                    video_name
                )
                record_policy(env, policy,
                              max_horizon=horizon,
                              task=subtask,
                              stochastic=args.stochastic,
                              q_fcn=qf,
                              video_name=video_name,
                              )
                n_rollouts += 1

            if args.option.lower() == 're':
                args.option = 'q'
                infinite_loop = True

        elif args.option.lower() == 'n':
            plots.plot_navitation2d()

        elif args.option.lower() == 'ie':
            args.option = 'e'
            infinite_loop = True
        else:
            print("Wrong option!")

        if not infinite_loop:
            args.option = None

