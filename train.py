import argparse
from pathlib import Path
import sys

import envs
from hiu_sac import HIUSAC

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))
from networks import MultiPolicyNet, MultiQNet

# Log and model saving parameters
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=None,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--env', '-e', type=str, default='navigation2d',
                    help='Name of environment [default: navigation2d]')
parser.add_argument('--log_dir', '-l', type=str, default=None,
                    help='Log directory [default: ./logs]')
parser.add_argument('--iterations', '-i', type=int, default=100,
                    help='Training iterations [default: 10]')
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


if __name__ == '__main__':

    # Parse and print out parameters
    args = parser.parse_args()

    env = get_environment(args.env, args.task, args.seed, args.render)

    algo = HIUSAC(
        env,
        net_size=64,
        use_q2=True,
        explicit_vf=False,
        train_rollouts=5,
        eval_rollouts=3,
        max_horizon=30,
        fixed_horizon=True,
        render=args.render,
        gpu_id=args.gpu,
        seed=args.seed,

        batch_size=64,
        replay_buffer_size=1e6,
    )

    algo.train(args.iterations)

    input('Press a key to close the script...')