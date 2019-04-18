#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
from hiu_sac import HIUSAC
from logger import setup_logger
from paper_environments import get_env


# Script parameters
def get_script_arguments():
    # Options
    env_name_choices = ['navigation2d', 'reacher', 'pusher', 'centauro']
    algo_choices = ['hiusac-p', 'hiusac', 'sac']

    parser = argparse.ArgumentParser(
        description='Train a policy with HIU-SAC algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--env_name', '-e', type=str,
                        default=env_name_choices[0],
                        help="Environment's name", choices=env_name_choices)
    parser.add_argument('--algo', type=str, default=algo_choices[0],
                        help='Algorithm name', choices=algo_choices)
    parser.add_argument('--episodes', '-i', type=int, default=None,
                        help="Number of algorithm episodes/iterations")
    parser.add_argument('--subtask', '-s', type=int, default=-1,
                        help="Environment subtask number")
    parser.add_argument('--log_dir', '-l', type=str, default="./training_logs",
                        help="Log directory")
    parser.add_argument('--snap_mode', type=str, default='gap_and_last',
                        help="Logging mode")
    parser.add_argument('--snap_gap', type=int, default=25,
                        help="Logging gap between iterations")
    parser.add_argument('--no_log_stdout', dest='no_log_stdout',
                        action='store_true',
                        help="Do not print logging messages in stdout.")
    parser.add_argument('--render', dest='render',
                        action='store_true',
                        help="Render environment during training")
    parser.add_argument('--seed', type=int, default=610, help="Seed value")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID")
    return parser.parse_args()


# Default HIU-SAC hyperparameters
def get_default_hiusac_hyperparams(env_name):
    if env_name.lower() == 'navigation2d':
        algo_hyperparams = {
            'nets_hidden_size': 64,
            'use_q2': True,
            'explicit_vf': False,
            'total_episodes': 100,
            'eval_rollouts': 3,
            'max_horizon': 30,
            'fixed_horizon': True,
            'render': False,
            'gpu_id': -1,
            'seed': 610,

            'batch_size': 64,
            'replay_buffer_size': 1e6,

            'i_entropy_scale': 1.,

            'auto_alpha': True,
            'i_tgt_entro': 0.e-1,
            'u_tgt_entros': None,

            'norm_input_pol': False,
            'norm_input_vfs': False,
        }
    elif env_name.lower() == 'reacher':
        algo_hyperparams = dict(
            nets_hidden_size=128,
            use_q2=True,
            explicit_vf=False,
            total_episodes=200,
            train_steps=2250,
            eval_rollouts=10,
            # max_horizon=1000,
            max_horizon=750,  # Con skip 50, desde 27-02 a las 18.21
            fixed_horizon=False,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            replay_buffer_size=5e6,

            i_entropy_scale=1.,

            auto_alpha=True,
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
            nets_hidden_size=128,
            use_q2=True,
            explicit_vf=False,
            total_episodes=200,
            train_steps=3000,
            eval_rollouts=10,
            # max_horizon=5000,  # Con skip 10
            # max_horizon=1500,  # Con skip 50
            # max_horizon=300,  # Con skip 50, desde 27-02 a las 17.50
            max_horizon=1000,  # Con skip 50, desde 27-02 a las 17.50
            fixed_horizon=False,
            render=False,
            gpu_id=-1,
            seed=610,

            batch_size=256,
            replay_buffer_size=5e6,

            i_entropy_scale=1.,

            auto_alpha=True,
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
            nets_hidden_size=256,
            use_q2=True,
            explicit_vf=False,
            total_episodes=300,
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
    args = get_script_arguments()

    if args.subtask == -1:
        args.subtask = None

    # Get Environment
    env, env_params = get_env(env_name=args.env_name, subtask=args.subtask,
                              seed=args.seed, render=args.render)

    # Get default algorithm hyperparameters
    algo_hyperparams = get_default_hiusac_hyperparams(args.env_name)

    # Replacing default hyperparameters with user arguments
    if args.algo.lower() == 'hiusac':
        algo_hyperparams['multitask'] = True
        algo_hyperparams['combination_method'] = 'convex'
    elif args.algo.lower() == 'hiusac-p':
        algo_hyperparams['multitask'] = True
        algo_hyperparams['combination_method'] = 'product'
    elif args.algo.lower() == 'hiusac-m':
        algo_hyperparams['multitask'] = True
        algo_hyperparams['combination_method'] = 'gmm'
    elif args.algo.lower() == 'sac':
        algo_hyperparams['multitask'] = False
    else:
        raise NotImplementedError("Algorithm option %s not available!"
                                  % args.algo)

    algo_hyperparams['render'] = args.render
    algo_hyperparams['gpu_id'] = args.gpu
    algo_hyperparams['seed'] = args.seed
    if args.episodes is not None:
        algo_hyperparams['total_episodes'] = args.episodes

    # Experiment variant. The corresponding values will be in variant.json file
    expt_variant = {
        'algo_name': args.algo.lower(),
        'algo_params': algo_hyperparams,
        'env_name': args.env_name,
        'env_params': env_params,
    }

    # Configure the logger: logging options and the log files/directories
    log_dir = setup_logger(
        exp_prefix=args.env_name + '-' + args.algo.lower(),
        seed=args.seed,
        variant=expt_variant,
        snapshot_mode=args.snap_mode,
        snapshot_gap=args.snap_gap,
        log_dir=args.log_dir,
        log_stdout=not args.no_log_stdout,
    )

    print("The log directory for the training process is: %s" % log_dir)

    # Instantiate HIU-SAC algorithm (or single-task SAC algorithm)
    algo = HIUSAC(env, **algo_hyperparams)

    # Training process
    expected_accum_rewards = algo.train()

    # Plot the expected accum. rewards obtained during the learning process
    plt.plot(expected_accum_rewards)
    plt.show(block=False)
    plt.savefig(args.algo.lower() + '_expected_accum_rewards.png')

    if not args.no_log_stdout:
        print("Closing the script. Bye!")

