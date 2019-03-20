import argparse
from pathlib import Path
import sys

from hiu_sac import HIUSAC
from logger.logger import setup_logger
from envs import get_normalized_env

# Add local files to path
root_dir = Path.cwd()
sys.path.append(str(root_dir))

# Script parameters
parser = argparse.ArgumentParser(description='Train Arguments')
parser.add_argument('--seed', '-s', type=int, default=610,
                    help='Seed value [default: 610]')
parser.add_argument('--task', '-t', type=int, default=-1,
                    help='Task number [default: None (Main task)]')
parser.add_argument('--env', '-e', type=str, default='navigation2d',
                    help='Name of environment [default: navigation2d]')
parser.add_argument('--log_dir', '-l', type=str, default=None,
                    help='Log directory [default: ./logs]')
parser.add_argument('--snap_mode', type=str, default='gap_and_last')
parser.add_argument('--snap_gap', type=int, default=25)
parser.add_argument('--iterations', '-i', type=int, default=None,
                    help='Training iterations '
                         '[default: None (recommended number of iterations)]')
parser.add_argument('--render', '-r', dest='render', default=False,
                    action='store_true',
                    help='Render environment during training [default: False]')
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID [default: -1 (cpu)]')
parser.add_argument('--algo', type=str, default='hiusac',
                    help='GPU ID [default: -1 (cpu)]')


def get_default_hiusac_hyperparams(env_name):
    if env_name.lower() == 'navigation2d':
        algo_hyperparams = dict(
            net_size=64,
            use_q2=True,
            explicit_vf=False,
            total_iterations=100,
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
            i_tgt_entro=0.e-1,
            u_tgt_entros=None,

            norm_input_pol=False,
            norm_input_vfs=False,
        )
    elif env_name.lower() == 'reacher':
        algo_hyperparams = dict(
            net_size=128,
            use_q2=True,
            explicit_vf=False,
            total_iterations=200,
            # train_rollouts=3,
            train_steps=750,
            eval_rollouts=5,
            # max_horizon=1000,
            max_horizon=150,  # Con skip 50, desde 27-02 a las 18.21
            fixed_horizon=False,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            replay_buffer_size=5e6,

            i_entropy_scale=1.,

            auto_alpha=True,
            # auto_alpha=False,
            # i_tgt_entro=None,
            # i_tgt_entro=1.e-0,  # Hasta 23-02
            # i_tgt_entro=1.e-0,
            # u_tgt_entros=[0.e-0, 0.e-0],
            i_tgt_entro=1.e-0,
            u_tgt_entros=None,

            norm_input_pol=False,
            norm_input_vfs=False,
        )
    elif env_name.lower() == 'pusher':
        algo_hyperparams = dict(
            net_size=128,
            use_q2=True,
            explicit_vf=False,
            total_iterations=500,
            # train_rollouts=2,
            train_steps=750,
            eval_rollouts=5,
            # max_horizon=5000,  # Con skip 10
            # max_horizon=1500,  # Con skip 50
            max_horizon=300,  # Con skip 50, desde 27-02 a las 17.50
            fixed_horizon=False,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            replay_buffer_size=5e6,

            i_entropy_scale=1.,

            auto_alpha=True,
            # auto_alpha=False,
            # i_tgt_entro=2.e-0,  # EN paper
            i_tgt_entro=1.e-0,  # Nuevo - final en paper
            # i_tgt_entro=5.e-1,  # Prueba 05/03   16:33
            # i_tgt_entro=1.e-1,  # Prueba 05/03   16:30
            u_tgt_entros=None,

            norm_input_pol=False,
            norm_input_vfs=False,
        )
    elif env_name.lower() == 'centauro':
        algo_hyperparams = dict(
            net_size=256,
            use_q2=True,
            explicit_vf=False,
            total_iterations=300,
            # train_rollouts=5,
            train_steps=5000,
            eval_rollouts=5,
            max_horizon=1000,
            # max_horizon=10,
            fixed_horizon=False,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            # replay_buffer_size=5e6,
            replay_buffer_size=10e6,

            i_entropy_scale=1.,

            auto_alpha=True,
            # auto_alpha=False,
            # i_tgt_entro=0.e+0,  # Exp 24/02
            # u_tgt_entros=[0.e+0, 0e+0],
            i_tgt_entro=1.e+0,
            # u_tgt_entros=[1.e+0, -2e0],
            u_tgt_entros=[1.e+0, 1e+0],  # Desde 27-02 a las 13:00

            norm_input_pol=False,
            norm_input_vfs=False,
        )
    else:
        raise ValueError("Wrong environment name '%s'" % env_name)

    return algo_hyperparams


if __name__ == '__main__':
    # Parse and print out parameters
    args = parser.parse_args()

    if args.task == -1:
        args.task = None

    # Get Environment
    env, env_params = get_normalized_env(args.env, args.task, args.seed, args.render)

    # Get default algorithm hyperparameters
    default_hyperparams = get_default_hiusac_hyperparams(args.env)
    if args.algo.lower() == 'hiusac':
        default_hyperparams['multitask'] = True
        default_hyperparams['combination_method'] = 'convex'
    elif args.algo.lower() == 'hiusac-p':
        default_hyperparams['multitask'] = True
        default_hyperparams['combination_method'] = 'product'
    elif args.algo.lower() == 'hiusac-m':
        default_hyperparams['multitask'] = True
        default_hyperparams['combination_method'] = 'gmm'
    elif args.algo.lower() == 'sac':
        default_hyperparams['multitask'] = False
    else:
        raise NotImplementedError("Option %s for algorithm not available"
                                  % args.algo)

    # Replacing default hyperparameters
    default_hyperparams['render'] = args.render
    default_hyperparams['gpu_id'] = args.gpu
    default_hyperparams['seed'] = args.seed
    if args.iterations is not None:
        default_hyperparams['total_iterations'] = args.iterations

    expt_variant = dict(
        algo_name=args.algo.lower(),
        algo_params=default_hyperparams,
        env_name=args.env,
        env_params=env_params,
    )

    log_dir = setup_logger(
        exp_prefix=args.env + '-' + args.algo.lower(),
        seed=args.seed,
        variant=expt_variant,
        snapshot_mode=args.snap_mode,
        snapshot_gap=args.snap_gap,
        log_dir=args.log_dir,
    )
    algo = HIUSAC(
        env,
        **default_hyperparams
    )

    algo.train(init_iteration=0)

    print("Closing the script. Bye!")

