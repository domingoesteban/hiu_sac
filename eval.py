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
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--env', '-e', type=str, default='navigation2d',
                    help='Name of environment [default: navigation2d]')
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


def eval_policy(env_name, models_directory, task=None, seed=610, gpu_id=-1):
    env = get_environment(env_name, args.task, args.seed, render=True)
    # Get Policy
    policy_file = osp.join(models_directory, 'policy.pt')
    policy = torch.load(policy_file)

    while True:
        input("Press a key to sample from the environment")
        rollout(env, policy.cpu(), max_horizon=1000, device='cpu')


if __name__ == '__main__':

    # Parse and print out parameters
    args = parser.parse_args()


    # Plot training data
    log_dir = args.log_dir
    progress_file = osp.join(log_dir, 'progress.csv')
    plot_progress(progress_file)

    # Evaluate policy in the environment
    # test_policy = False
    test_policy = True
    if test_policy:
        models_dir = osp.join(log_dir, 'models', 'last_itr')
        eval_policy(args.env, models_dir, args.task, args.seed, args.gpu)


    input('Press a key to close the script...')