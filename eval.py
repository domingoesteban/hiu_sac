import os.path as osp
import numpy as np
import torch
import argparse
import json


from paper_environments import get_env
from utils import rollout
import plots


VIDEOS_DIR = 'videos'


# Numpy print options
np.set_printoptions(precision=3, suppress=True)

# # Script options
# options_dict = dict(
#     e="evaluate",
#     ie="evaluate infinite times!",
#     p="plot progress",
#     v="plot qval",
#     r="record interaction",
#     re="record and exit",
#     n="navigation2d",
#     t="Change policy task",
#     i="Change iteration",
#     et="change env task",
#     h="change evaluation horizon",
#     q="exit this script",
# )
options_choices = [
    ('e', 'evaluate', "Evaluate the policy"),
    ('p', 'plot', "Plot the expected return from the learning process"),
    ('pi', 'plot_info', "Plot relevant information from the learning process"),
    ('er', 'eval_repeat', "Evaluate repeteadly the environment!"),
]

# Script parameters
parser = argparse.ArgumentParser(
    description='Evaluate a policy from a log directory.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
exclusive_opts = parser.add_mutually_exclusive_group()
for (short_opt, long_opt, description) in options_choices:
    exclusive_opts.add_argument('-'+short_opt, '--'+long_opt, help=description,
                                action='store_const', dest='script_option',
                                const=short_opt)
parser.add_argument('log_dir', type=str, help='Full path of the log directory')
parser.add_argument('--seed', '-s', type=int, default=610, help='Seed value')
parser.add_argument('--pol_task', '-pt', type=int, default=None,
                    help="Policy task number. None is the Main Task")
parser.add_argument('--env_task', '-et', type=int, default=None,
                    help="Environment task number. None is the Main Task")
parser.add_argument('--horizon', '-n', type=int, default=None,
                    help="Rollout horizon. None used the max_horizon parameter"
                         "value from the log directory.")
parser.add_argument('--iteration', '-i', type=int, default=-1,
                    help="Model iteration. -1 for the last available episode")
parser.add_argument('--gpu', type=int, default=-1,
                    help='GPU ID. For using the cpu selects -1')
parser.add_argument('--stochastic', action='store_true')
# By default the script evaluates the last policy
parser.set_defaults(script_option='e')

n_rollouts = 0
environment = None


def plot_progress(progress_file, algo_name='hiusac', only_expect_return=True):
    """Plot relevant data from the training process logged in a file.
    Args:
        progress_file (str): Full path of the progress file.
        algo_name (str): Algorithm name.

    Returns:
        None

    """
    if algo_name in ['hiusac', 'hiusac-p']:
        num_intentions = 2
    else:
        num_intentions = None
    plots.plot_intentions_eval_returns(
        progress_file,
        num_intentions=num_intentions,
    )
    if not only_expect_return:
        plots.plot_intentions_info(
            progress_file,
            num_intentions=num_intentions,
        )


def eval_policy(env, policy, max_horizon=50, task=None, stochastic=False):
    """Evaluate a policy in a specific environment.

    Args:
        env (Env): Environment
        policy (torch.nn.Module): Policy
        max_horizon (int): Maximum horizon
        task (int or None): Policy subtask
        stochastic (bool): Select actions by sampling from the policy

    Returns:
        None

    """

    rollout_info = rollout(
        env,
        policy,
        max_horizon=max_horizon,
        fixed_horizon=False,
        device='cpu',
        render=True,
        return_info_dict=True,
        record_video_name=None,
        intention=task,
        deterministic=not stochastic,
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
        return_info_dict=False,
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

    # the full path of the log directory
    log_dir = args.log_dir

    # Get experiment variant data from log directory
    with open(osp.join(log_dir, 'variant.json')) as json_data:
        log_data = json.load(json_data)
        env_name = log_data['env_name']
        log_env_params = log_data['env_params']
        algo_params = log_data['algo_params']
        seed = algo_params['seed']
        horizon = algo_params['max_horizon']

    progress_file = None
    env = None
    policy = None
    env_params = None
    run_once = True
    models_dir = None
    qf = None

    # Get models directory
    itr_dir = 'itr_%03d' % args.iteration if args.iteration > -1 else 'last_itr'
    models_dir = osp.join(log_dir, 'models', itr_dir)

    # Get the progress file
    if args.script_option in ['p', 'pi']:
        if progress_file is None:
            progress_file = osp.join(log_dir, 'progress.csv')

    # Get environment and policy
    if args.script_option in ['e', 'er', 'v']:
        env, env_params = get_env(
            env_name, args.pol_task, args.seed, render=True,
            new_env_params=log_env_params
        )

        pol_file = osp.join(models_dir, 'policy.pt')
        policy = torch.load(pol_file, map_location=lambda storage, loc: storage)

        if args.horizon is not None:
            horizon = args.horizon

    # Get Q-value function
    if args.script_option in ['v']:
        qf_file = osp.join(models_dir, 'qf1.pt')
        qf = torch.load(qf_file, map_location=lambda storage, loc: storage)

    # Run the script with the selected option
    if args.script_option == 'e':
        eval_policy(env, policy,
                    max_horizon=horizon,
                    task=args.pol_task,
                    stochastic=args.stochastic,
                    )

    elif args.script_option == 'er':
        try:
            while True:
                eval_policy(env, policy,
                            max_horizon=horizon,
                            task=args.pol_task,
                            stochastic=args.stochastic,
                            )
        except KeyboardInterrupt:
            pass

    elif args.script_option == 'p':
        plot_progress(progress_file, log_data['algo_name'],
                      only_expect_return=True)

    elif args.script_option == 'pi':
        plot_progress(progress_file, log_data['algo_name'],
                      only_expect_return=False)

    elif args.script_option in ['r', 're']:
        max_iter = 300
        max_rollouts = 10
        # range_list = list(range(0, max_iter, 25)) + [None]
        range_list = [args.iteration]

        env_subtask = None if args.env_task == -1 else args.env_task
        env.set_active_subtask(env_subtask)

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
                break

    elif args.script_option == 'v':
        obs = np.zeros(env.obs_dim)
        # TODO: Make this for all envs
        obs[0] = -6
        obs[1] = -6
        plot_value_fcn(qf, policy, obs, env.action_space.low, env.action_space.high)

    # # Menu option
    # print("Available options:")
    # for arg in vars(args):
    #     print('\t%s: %s' % (arg, getattr(args, arg)))
    #     if arg == 'script_option':
    #         for (option, _, description) in options_choices:
    #             print("\t  %s: %s" % (option, description))
    # print('or')
    # print('\t%s: %s' % ('q', "Close the script"))
    # user_option = input('Select an option:')

    if args.script_option in ['p', 'pi']:
        input('Press a key to close the script')
    else:
        print("Closing the script. Bye!")
