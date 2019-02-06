import argparse
from pathlib import Path
import sys
import os.path as osp
import torch
import json


import envs
from envs import get_normalized_env
from hiu_sac import HIUSAC
from utils import interaction
from utils import rollout
import plots

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
parser.add_argument('--horizon', '-o', type=int, default=None,
                    help='Rollout horizon [default: 100]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')
parser.add_argument('--stochastic', action='store_true')


def plot_progress(progress_file):
    """
    Function for plotting useful data from a learning process.
    :param progress_file:
    :return:
    """
    plots.plot_intentions_eval_returns(progress_file)
    plots.plot_intentions_info(progress_file)


def eval_policy(env, policy, max_horizon=50, task=None, stochastic=False):
    rollout(
        env, policy,
        max_horizon=max_horizon,
        fixed_horizon=False,
        device='cpu',
        render=True,
        intention=task, deterministic=not stochastic,
    )


def plot_value_fcn(qf, policy, env):
    import numpy as np
    obs = np.zeros(env.obs_dim)
    actions_dims = (0, 1)
    obs[actions_dims[0]] = -2
    obs[actions_dims[1]] = -2

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
    models_dir = osp.join(log_dir, 'models', 'last_itr')
    policy_file = osp.join(models_dir, 'policy.pt')
    qf_file = osp.join(models_dir, 'qf1.pt')
    policy = torch.load(policy_file).cpu()
    qf = torch.load(qf_file).cpu()

    while True:
        user_input = input("Select an option "
                           "("
                           "'p':plot progress, "
                           "'e':evaluate, "
                           "'v':plot_qval,"
                           "'t':change policy task,"
                           "'et':change env task,"
                           "). "
                           "Or 'q' to exit: ")
        # user_input = 'e'
        if user_input.lower() == 'q':
            print("Closing the script. Bye!")
            break
        elif user_input.lower() == 'p':
            plot_progress(progress_file)
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
            new_task = int(new_task)
            if new_task not in list(range(-1, env.n_subgoals)):
                print("Wrong option '%s'!" % new_task)
            new_task = None if new_task == -1 else new_task
            env.set_subtask(new_task)
        elif user_input.lower() == 'e':
            if args.horizon is not None:
                horizon = args.horizon
            eval_policy(env, policy,
                        max_horizon=horizon,
                        task=args.task,
                        stochastic=args.stochastic,
                        )
        else:
            print("Wrong option!")

