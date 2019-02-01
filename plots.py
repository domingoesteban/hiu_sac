import sys
import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import traceback


def get_csv_data(csv_file, labels, space_separated=False):
    data, all_labels = get_csv_data_and_labels(csv_file,
                                               space_separated=space_separated)

    for label in all_labels:
        print(label)
    print('***\n'*3)
    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            try:
                new_data[ll, :] = data[:, idx]
            except Exception:
                print(traceback.format_exc())
                print("Error with data in %s" % csv_file)
                sys.exit(1)
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file, space_separated=False):
    # Read from CSV file
    try:
        if space_separated:
            series = pd.read_csv(csv_file, delim_whitespace=True)
        else:
            series = pd.read_csv(csv_file)
    except Exception:
        print(traceback.format_exc())
        print("Error reading %s" % csv_file)
        sys.exit(1)

    data = series.as_matrix()
    labels = list(series)

    return data, labels


def set_latex_plot():
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # rc('font', **{'family': 'serif','serif':['Times']})
    matplotlib.rcParams['font.family'] = ['serif']
    matplotlib.rcParams['font.serif'] = ['Times New Roman']


def subplots(*args, **kwargs):
    fig, axs = plt.subplots(*args, **kwargs)

    if isinstance(axs, np.ndarray):
        for aa in axs:
            axis_format(aa)
    else:
        axis_format(axs)

    return fig, axs


def fig_format(fig):
    fig.subplots_adjust(hspace=0)
    fig.set_facecolor((1, 1, 1))


def axis_format(axis):
    # axis.tick_params(axis='x', labelsize=25)
    # axis.tick_params(axis='y', labelsize=25)
    axis.tick_params(axis='x', labelsize=15)
    axis.tick_params(axis='y', labelsize=15)

    # Background
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.xaxis.grid(color='white', linewidth=2)
    axis.set_facecolor((0.917, 0.917, 0.949))


def plot_contours(ax, values, x_min, x_max, y_min, y_max, delta=0.05):
    # xlim = (1.1*x_min, 1.1*x_max)
    # ylim = (1.1*y_min, 1.1*y_max)
    xlim = (1.0*x_min, 1.0*x_max)
    ylim = (1.0*y_min, 1.0*y_max)
    all_x = np.arange(x_min, x_max, delta)
    all_y = np.arange(y_min, y_max, delta)
    xy_mesh = np.meshgrid(all_x, all_y)
    values = values.reshape(len(all_x), len(all_y))

    contours = ax.contour(xy_mesh[0], xy_mesh[1], values, 20,
                          colors='dimgray')
    ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
    ax.imshow(values, extent=(x_min, x_max, y_min, y_max), origin='lower',
              alpha=0.5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('Vel. X', fontweight='bold', fontsize=18)
    ax.set_ylabel('Vel. Y', fontweight='bold', fontsize=18)
    ax.axis('equal')
    ax.set_aspect('equal', 'box')
    ax.grid(False)


def plot_q_fcn(i_qf, i_qf2, u_qf, u_qf2, obs, policy):
    # Load environment
    dirname = os.path.dirname(args.file)
    with open(os.path.join(dirname, 'variant.json')) as json_data:
        env_params = json.load(json_data)['env_params']

    env = NormalizedBoxEnv(
        Navigation2dGoalCompoEnv(**env_params),
        # normalize_obs=True,
        normalize_obs=False,
        online_normalization=False,
        obs_mean=None,
        obs_var=None,
        obs_alpha=0.001,
    )
    # env.reset()
    # env.render()

    obs = np.array(obs)
    n_action_samples = 100
    x_min, y_min = env.action_space.low
    x_max, y_max = env.action_space.high
    delta = 0.05
    # xlim = (1.1*x_min, 1.1*x_max)
    # ylim = (1.1*y_min, 1.1*y_max)
    xlim = (1.0*x_min, 1.0*x_max)
    ylim = (1.0*y_min, 1.0*y_max)
    all_x = np.arange(x_min, x_max, delta)
    all_y = np.arange(y_min, y_max, delta)
    xy_mesh = np.meshgrid(all_x, all_y)

    all_acts = np.zeros((len(all_x)*len(all_y), 2))
    all_acts[:, 0] = xy_mesh[0].ravel()
    all_acts[:, 1] = xy_mesh[1].ravel()

    n_unintentions = u_qf.n_heads if u_qf is not None else 0

    def plot_action_samples(ax, actions):
        x, y = actions[:, 0], actions[:, 1]
        ax.scatter(x, y, c='b', marker='*', zorder=5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    for ob in obs:
        all_obs = np.broadcast_to(ob, (all_acts.shape[0], 2))

        fig, all_axs = \
            subplots(1, n_unintentions + 1,
                         gridspec_kw={'wspace': 0, 'hspace': 0},
        )
        # fig.suptitle('Q-val Observation: ' + str(ob))
        fig.tight_layout()
        fig.canvas.set_window_title('q_vals_%1d_%1d' % (ob[0], ob[1]))

        all_axs = np.atleast_1d(all_axs)

        all_axs[0].set_title('Main Task',
                             fontdict={'fontsize': 30, 'fontweight': 'medium'})
        q_vals = i_qf.get_values(all_obs, all_acts)[0]
        if i_qf2 is not None:
            q2_vals = i_qf2.get_values(all_obs, all_acts)[0]
            q_vals = np.concatenate([q_vals, q2_vals], axis=1)
            q_vals = np.min(q_vals, axis=1, keepdims=True)

        plot_contours(all_axs[0],
                      q_vals,
                      x_min, x_max,
                      y_min, y_max,
                      delta=0.05)

        if u_qf is None:
            pol_kwargs = dict(
            )
        else:
            pol_kwargs = dict(
                pol_idx=None,
            )

        # Compute and plot Main Task Q Value
        action_samples = policy.get_actions(all_obs[:n_action_samples, :],
                                            deterministic=False,
                                            **pol_kwargs
                                            )[0]
        plot_action_samples(all_axs[0], action_samples)
        all_axs[0].set_xticklabels([])
        all_axs[0].set_yticklabels([])

        for aa in range(n_unintentions):
            subgo_ax = all_axs[aa + 1]
            subgo_ax.set_title('Sub-Task %02d' % (aa+1),
                               fontdict={'fontsize': 30, 'fontweight': 'medium'} )

            q_vals = u_qf.get_values(all_obs, all_acts, val_idxs=[aa])[0]
            q_vals = q_vals[0]

            if u_qf2 is not None:
                q2_vals = u_qf2.get_values(all_obs, all_acts)[0]
                q2_vals = q2_vals[0]
                q_vals = np.concatenate([q_vals, q2_vals], axis=1)
                q_vals = np.min(q_vals, axis=1, keepdims=True)

            plot_contours(subgo_ax, q_vals)

            if u_qf is None:
                pol_kwargs = dict(
                )
            else:
                pol_kwargs = dict(
                    pol_idx=aa,
                )

            # Compute and plot Sub-Task Q Value
            action_samples = policy.get_actions(all_obs[:n_action_samples, :],
                                                deterministic=False,
                                                **pol_kwargs
                                                )[0]
            plot_action_samples(subgo_ax, action_samples)

            subgo_ax.get_yaxis().set_visible(False)
            subgo_ax.set_xticklabels([])

        all_axs[0].set_xticklabels([])
        all_axs[0].set_yticklabels([])

    # plt.subplots_adjust(wspace=0, hspace=0)


def get_csv_data(csv_file, labels, space_separated=False):
    data, all_labels = get_csv_data_and_labels(csv_file,
                                               space_separated=space_separated)

    for label in all_labels:
        print(label)
    print('***\n'*3)
    n_data = data.shape[0]

    new_data = np.zeros((len(labels), n_data))

    # # Uncomment for debugging
    # print(all_labels)

    for ll, name in enumerate(labels):
        if name in all_labels:
            idx = all_labels.index(name)
            try:
                new_data[ll, :] = data[:, idx]
            except Exception:
                print(traceback.format_exc())
                print("Error with data in %s" % csv_file)
                sys.exit(1)
        else:
            raise ValueError("Label '%s' not available in file '%s'"
                             % (name, csv_file))

    return new_data


def get_csv_data_and_labels(csv_file, space_separated=False):
    # Read from CSV file
    try:
        if space_separated:
            series = pd.read_csv(csv_file, delim_whitespace=True)
        else:
            series = pd.read_csv(csv_file)
    except Exception:
        print(traceback.format_exc())
        print("Error reading %s" % csv_file)
        sys.exit(1)

    data = series.as_matrix()
    labels = list(series)

    return data, labels


def plot_intentions_eval_returns(csv_file, num_intentions=None, block=False):
    labels_to_plot = ['Test Returns Mean']

    if num_intentions is None:
        num_intentions = 0
    else:
        num_intentions += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(num_intentions):
            new_string = ('[U-%02d] ' % uu) + label
            new_labels.append(new_string)

        # Assuming the Main does not have a prefix
        new_string = label
        new_labels.append(new_string)

    n_subplots = len(labels_to_plot) * (num_intentions + 1)

    data = get_csv_data(csv_file, new_labels)

    fig, axs = subplots(n_subplots)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])
    fig.subplots_adjust(hspace=0)
    fig.suptitle('Average Return', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def plot_q_values(qf, policy=None, obs=None, action_dims=(0, 1), delta=0.05,
                  device='cpu'):

    # Values Plots
    ob = [-2., -2]
    num_intentions = 2

    action_dim_x = action_dims[0]
    action_dim_y = action_dims[1]

    x_min = env.action_space.low[action_dim_x]
    y_min = env.action_space.low[action_dim_y]
    x_max = env.action_space.high[action_dim_x]
    y_max = env.action_space.high[action_dim_y]

    all_x = torch.arange(x_min, x_max, delta)
    all_y = torch.arange(y_min, y_max, delta)
    xy_mesh = torch.meshgrid(all_x, all_y)

    all_acts = torch.zeros((len(all_x)*len(all_y), 2))
    all_acts[:, 0] = xy_mesh[0].contiguous().view(-1)
    all_acts[:, 1] = xy_mesh[1].contiguous().view(-1)

    fig, all_axs = \
        subplots(1, num_intentions + 1,
                 gridspec_kw={'wspace': 0, 'hspace': 0},
                 )
    # fig.suptitle('Q-val Observation: ' + str(ob))
    fig.tight_layout()
    fig.canvas.set_window_title('q_vals_%1d_%1d' % (ob[0], ob[1]))

    all_axs = np.atleast_1d(all_axs)

    all_axs[-1].set_title('Main Task',
                          fontdict={'fontsize': 30, 'fontweight': 'medium'})

    all_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    all_obs = all_obs.unsqueeze(0).expand_as(all_acts)

    q_vals = qf(all_obs, all_acts)
    for intention in range(num_intentions + 1):
        ax = all_axs[intention]
        plot_contours(ax, q_vals[:, intention, :].cpu().data.numpy(),
                      x_min, x_max, y_min, y_max, delta=delta)

        # Plot action samples
        #
        # x, y = actions[:, 0], actions[:, 1]
        # ax.scatter(x, y, c='b', marker='*', zorder=5)
        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        if intention < num_intentions:
            ax.set_title('Sub-Task %02d' % (intention+1),
                         fontdict={'fontsize': 30,
                                   'fontweight': 'medium'}
                         )

    plt.show()
