import numpy as np
import argparse
from pathlib import Path
import sys
import os.path as osp
import torch
import json
import pybullet as pb


from envs import get_normalized_env
from utils import rollout
import plots

# Numpy print options
np.set_printoptions(precision=3, suppress=True)

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))

# Script parameters
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('log_dir', type=str,
                    help='Log directory')
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--horizon', '-n', type=int, default=None,
                    help='Rollout horizon [default: 100]')
parser.add_argument('--iteration', '-i', type=int, default=None,
                    help='Model iteration [default: last]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')
parser.add_argument('--stochastic', action='store_true')
parser.add_argument('--option', '-o', type=str, default=None,
                    help='Script option [default: None]')


def plot_progress(progress_file, algo_name='hiusac'):
    """
    Function for plotting useful data from a learning process.
    :param progress_file:
    :return:
    """
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
        env, policy,
        max_horizon=max_horizon,
        fixed_horizon=False,
        device='cpu',
        render=True,
        intention=task, deterministic=not stochastic,
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
                  q_fcn=None, video_name='hiu_sac'):
    # video_name = 'temporal.mp4'
    video_dir = 'videos'

    video_name = osp.join(
        video_dir,
        video_name + '.mp4',
    )

    rollout_info = rollout(
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


def plot_value_fcn(qf, policy, env):
    obs = np.zeros(env.obs_dim)
    actions_dims = (0, 1)

    obs[actions_dims[0]] = -6
    obs[actions_dims[1]] = -6

    plots.plot_q_values(
        qf,
        action_lower=env.action_space.low,
        action_higher=env.action_space.high,
        obs=obs,
        policy=policy,
        action_dims=actions_dims,
        delta=0.05,
        device='cpu'
    )


def plot_navitation2d():
    from envs import get_normalized_env
    env, env_params = get_normalized_env(
        'navigation2d',
        None,
        610,
        False
    )
    env.render()

    colors = np.array([
        'red',
        'green',
        'blue',
        'black',
        'purple',
    ])

    obs = [
        (-2., -2.),
        (-2., 4.),
        (4., -2.),
        (4., 4.),
        (-6., -6.),
    ]

    for ob, color in zip(obs, colors):
        env._wrapped_env._robot_marker(
            env._wrapped_env._main_ax,
            ob[0],
            ob[1],
            color=color,
            zoom=0.03
        )

    input('cucucu')


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
    itr_dir = 'itr_%03d' % args.iteration if args.iteration is not None else 'last_itr'
    models_dir = osp.join(log_dir, 'models', 'last_itr')
    policy_file = osp.join(models_dir, 'policy.pt')
    qf_file = osp.join(models_dir, 'qf1.pt')
    policy = torch.load(policy_file, map_location=lambda storage, loc: storage)
    qf = torch.load(qf_file, map_location=lambda storage, loc: storage)

    infinite_loop = False
    first_time = True
    while True:
        if not infinite_loop:
            if not first_time or args.option is None:
                user_input = input("Select an option: \n"
                                   "\t'p':plot progress\n"
                                   "\t'v':plot_qval\n"
                                   "\t't':change policy task\n"
                                   "\t'i':change iteration\n"
                                   "\t'et':change env task\n"
                                   "\t'h':change evaluation horizon\n"
                                   "\t'e':evaluate\n"
                                   "\t'r':record interaction\n"
                                   "\t'n':navigation2d\n"
                                   "\t'q' to exit\n"
                                   "Option: ")
            else:
                user_input = args.option
                first_time = False
        # user_input = 'e'
        if user_input.lower() == 'q':
            print("Closing the script. Bye!")
            break

        elif user_input.lower() == 'p':
            plot_progress(progress_file, log_data['algo_name'])

        elif user_input.lower() == 'v':
            plot_value_fcn(qf, policy, env)

        elif user_input.lower() == 't':
            new_task = input("Specify task id (-1 for None). Task id: ")
            new_task = int(new_task)
            if new_task not in list(range(-1, policy.num_intentions)):
                print("Wrong option '%s'!" % new_task)
            args.task = None if new_task == -1 else new_task
            print("New task is %d" % new_task)

        elif user_input.lower() == 'et':
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

        elif user_input.lower() == 'h':
            new_horizon = input("Specify new horizon: ")
            new_horizon = int(new_horizon)
            if not new_horizon > 1:
                print("Wrong horizon '%d'!" % new_horizon)
            else:
                args.horizon = new_horizon
                print("New horizon is %d" % new_horizon)

        elif user_input.lower() == 'e':
            if args.horizon is not None:
                horizon = args.horizon
            eval_policy(env, policy,
                        max_horizon=horizon,
                        task=args.task,
                        stochastic=args.stochastic,
                        q_fcn=qf,
                        )
        elif user_input.lower() == 'r':
            if args.horizon is not None:
                horizon = args.horizon
            env_subtask = env.get_subtask()
            if env_subtask is None:
                env_subtask = -1
            if args.task is None:
                subtask = -1
            else:
                subtask = args.task
            video_name = (
                    env_name +
                    ('_s%03d' % seed) +
                    ('_task%01d' % subtask) +
                    ('_envtask%01d' % env_subtask)
            )
            record_policy(env, policy,
                          max_horizon=horizon,
                          task=subtask,
                          stochastic=args.stochastic,
                          q_fcn=qf,
                          video_name=video_name,
                          )

        elif user_input.lower() == 'n':
            plot_navitation2d()

        elif user_input.lower() == 'i':
            user_input = 'e'
            infinite_loop = True

        else:
            print("Wrong option!")

