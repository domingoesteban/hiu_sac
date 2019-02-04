import argparse
from pathlib import Path
import sys
import os.path as osp
import torch

import envs
from hiu_sac import HIUSAC
from utils import interaction
from utils import rollout
import plots

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))

# Log and model saving parameters
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('log_dir', type=str,
                    help='Log directory')
parser.add_argument('--env', '-e', type=str, default='navigation2d',
                    help='Name of environment [default: navigation2d]')
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--horizon', '-o', type=int, default=100,
                    help='Rollout horizon [default: 100]')
parser.add_argument('--render', '-r', dest='render', default=False,
                    action='store_true',
                    help='Render environment during training [default: False]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')


def get_environment(env_name, subtask=None, seed=610, render=False):
    if env_name.lower() not in [
        'navigation2d',
        'reacher'
    ]:
        raise ValueError("Wrong environment name '%s'" % env_name)
    print("Loading environment %s" % env_name)

    if env_name.lower() == 'navigation2d':
        environment = envs.Navitation2D(subtask=subtask, seed=seed)
    elif env_name.lower() == 'reacher':
        environment = envs.Reacher(subtask=subtask, seed=seed,
                                   render=render)

    return envs.NormalizedEnv(environment)


def plot_progress(progress_file):
    plots.plot_intentions_eval_returns(progress_file)
    plots.plot_intentions_info(progress_file)


def eval_policy(env, policy, max_horizon=50, task=None,
                seed=610, gpu_id=-1):
    rollout(
        env, policy,
        max_horizon=max_horizon,
        fixed_horizon=True,
        device='cpu',
        render=True,
        intention=task, deterministic=True,
    )


def plot_value_fcn(qf, policy, env):
    obs = (-2, -2)
    # obs = (4, 4)
    # obs = (-2, 4)
    plots.plot_q_values(
        qf,
        action_lower=env.action_space.low,
        action_higher=env.action_space.high,
        obs=obs,
        policy=policy,
        action_dims=(0, 1),
        delta=0.05,
        device='cpu'
    )


if __name__ == '__main__':

    # Parse and print out parameters
    args = parser.parse_args()

    # Plot training data
    log_dir = args.log_dir
    progress_file = osp.join(log_dir, 'progress.csv')

    # Get model
    env = get_environment(args.env, args.task, args.seed, render=True)

    # Get models from file
    models_dir = osp.join(log_dir, 'models', 'last_itr')
    policy_file = osp.join(models_dir, 'policy.pt')
    qf_file = osp.join(models_dir, 'qf1.pt')
    policy = torch.load(policy_file).cpu()
    qf = torch.load(qf_file).cpu()

    # Plot Q-values

    # Evaluate policy in the environment
    while True:
        user_input = input("Select an option "
                           "('p':plot progress, 'e':evaluate, 'v':plot_qval). "
                           "Or 'q' to exit: ")
        # user_input = 'v'
        if user_input.lower() == 'q':
            print("Closing the script. Bye!")
            break
        elif user_input.lower() == 'p':
            plot_progress(progress_file)
        elif user_input.lower() == 'v':
            plot_value_fcn(qf, policy, env)
        elif user_input.lower() == 'e':
            eval_policy(env, policy,
                        max_horizon=args.horizon,
                        task=args.task,
                        seed=args.seed,
                        gpu_id=args.gpu,
                        )
        else:
            print("Wrong option!")

