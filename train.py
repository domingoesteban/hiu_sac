import argparse
from pathlib import Path
import sys

import envs
from hiu_sac import HIUSAC

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))

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
parser.add_argument('--iterations', '-i', type=int, default=None,
                    help='Training iterations '
                         '[default: None (recommended number of iterations)]')
parser.add_argument('--render', '-r', dest='render', default=False,
                    action='store_true',
                    help='Render environment during training [default: False]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')


def get_environment(env_name, subtask=None, seed=610, render=False):
    print("Loading environment %s" % env_name)

    if env_name.lower() == 'navigation2d':
        environment = envs.Navitation2D(subtask=subtask, seed=seed)
    elif env_name.lower() == 'reacher':
        environment = envs.Reacher(subtask=subtask, seed=seed,
                                   render=render)
    elif env_name.lower() == 'pusher':
        environment = envs.Pusher(subtask=subtask, seed=seed,
                                  render=render)
    else:
        raise ValueError("Wrong environment name '%s'" % env_name)

    return envs.NormalizedEnv(environment)


def get_default_hiu_hyperparams(env_name):
    if env_name.lower() == 'navigation2d':
        algo_hyperparams = dict(
            net_size=64,
            use_q2=True,
            explicit_vf=False,
            total_iterations=50,
            train_rollouts=5,
            eval_rollouts=3,
            max_horizon=30,
            fixed_horizon=True,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=64,
            replay_buffer_size=1e6,

            i_entropy_scale=1.,

            auto_alpha=True,
            # auto_alpha=False,
            i_tgt_entro=None,
            u_tgt_entros=None,

        )
    elif env_name.lower() == 'reacher':
        algo_hyperparams = dict(
            net_size=128,
            use_q2=True,
            explicit_vf=False,
            total_iterations=200,
            train_rollouts=3,
            eval_rollouts=2,
            max_horizon=1000,
            fixed_horizon=True,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            replay_buffer_size=1e6,

            i_entropy_scale=1.,

            auto_alpha=True,
            # auto_alpha=False,
            i_tgt_entro=None,
            u_tgt_entros=None,
        )
    elif env_name.lower() == 'pusher':
        algo_hyperparams = dict(
            net_size=128,
            use_q2=True,
            explicit_vf=False,
            total_iterations=200,
            train_rollouts=1,
            eval_rollouts=2,
            max_horizon=1000,
            fixed_horizon=True,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=128,
            replay_buffer_size=1e6,

            i_entropy_scale=1.,

            # auto_alpha=True,
            auto_alpha=False,
            i_tgt_entro=None,
            u_tgt_entros=None,
        )
    else:
        raise ValueError("Wrong environment name '%s'" % env_name)

    return algo_hyperparams


if __name__ == '__main__':

    # Parse and print out parameters
    args = parser.parse_args()

    env = get_environment(args.env, args.task, args.seed, args.render)

    default_hyperparams = get_default_hiu_hyperparams(args.env)

    # Replacing default hyperparameters
    default_hyperparams['render'] = args.render
    default_hyperparams['gpu_id'] = args.gpu
    default_hyperparams['seed'] = args.seed
    if args.iterations is not None:
        default_hyperparams['total_iterations'] = args.iterations

    algo = HIUSAC(
        env,
        **default_hyperparams
    )

    algo.train(init_iteration=0)

    print("Closing the script. Bye!")

