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


def plot_intentions_info(csv_file, num_intentions=None, block=False):
    labels_to_plot = ['Alpha']

    if num_intentions is None:
        num_intentions = 0
    else:
        num_intentions += 1

    # Add Intentional-Unintentional Label
    new_labels = list()
    for label in labels_to_plot:
        for uu in range(num_intentions):
            new_string = label + ('[U-%02d] ' % uu)
            print(new_string)
            input('fdsaf')
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
    fig.suptitle('Alpha', fontweight='bold')

    for aa, ax in enumerate(axs):
        ax.plot(data[aa])
        ax.set_ylabel(new_labels[aa])
        plt.setp(ax.get_xticklabels(), visible=False)

    axs[-1].set_xlabel('Episodes')
    plt.setp(axs[-1].get_xticklabels(), visible=True)

    print('total_iters:', len(data[-1]))
    plt.show(block=block)


def plot_contours(ax, x_tensor, y_tensor, values):
    contours = ax.contour(x_tensor, y_tensor, values, 20,
                          colors='dimgray')
    ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
    ax.imshow(values, extent=(x_tensor.min(), x_tensor.max(),
                              y_tensor.min(), y_tensor.max()),
              origin='lower', alpha=0.5)


def plot_q_values(qf, action_lower, action_higher, obs, policy=None,
                  obs_dims=(0, 1), action_dims=(0, 1), delta=0.01,
                  device='cpu'):
    # Values Plots
    num_intentions = 2

    action_dim_x = action_dims[0]
    action_dim_y = action_dims[1]
    obs_dim_x = obs_dims[0]
    obs_dim_y = obs_dims[1]

    x_min = action_lower[action_dim_x]
    y_min = action_lower[action_dim_y]
    x_max = action_higher[action_dim_x]
    y_max = action_higher[action_dim_y]

    all_x = torch.arange(float(x_min), float(x_max), float(delta))
    all_y = torch.arange(float(y_min), float(y_max), float(delta))
    x_mesh, y_mesh = torch.meshgrid(all_x, all_y)

    x_mesh = x_mesh.t()
    y_mesh = y_mesh.t()
    all_acts = torch.stack((x_mesh, y_mesh), dim=-1)

    fig, all_axs = \
        subplots(1, num_intentions + 1,
                 gridspec_kw={'wspace': 0, 'hspace': 0},
                 )
    # fig.suptitle('Q-val Observation: ' + str(ob))
    fig.tight_layout()
    fig.canvas.set_window_title('q_vals_%1d_%1d' %
                                (obs[obs_dim_x], obs[obs_dim_y]))

    all_axs = np.atleast_1d(all_axs)

    all_axs[-1].set_title('Main Task',
                          fontdict={'fontsize': 30, 'fontweight': 'medium'})

    all_obs = torch.tensor(obs, dtype=torch.float32, device=device)
    all_obs = all_obs.expand_as(all_acts)

    q_vals = qf(all_obs, all_acts).squeeze(-1).detach()

    if policy is not None:
        n_samples = 50
        torch_obs = torch.tensor(obs, dtype=torch.float32, device=device)
        torch_obs = torch_obs.expand(n_samples, -1)
        actions, pol_info = policy(torch_obs,
                                   deterministic=False,
                                   intention=None)
        actions = actions.detach().numpy()
        intention_actions = pol_info['action_vect'].detach().numpy()

    for intention in range(num_intentions + 1):
        ax = all_axs[intention]
        plot_contours(ax, x_mesh, y_mesh, q_vals[:, :, intention])

        ax.set_xlabel('Vel. X', fontweight='bold', fontsize=18)
        ax.set_ylabel('Vel. Y', fontweight='bold', fontsize=18)
        ax.axis('equal')
        ax.set_aspect('equal', 'box')
        ax.grid(False)

        ax.set_xlim(x_min)
        ax.set_ylim(y_min)

        if policy is not None:
            if intention < num_intentions:
                x = intention_actions[:, intention, action_dim_x]
                y = intention_actions[:, intention, action_dim_y]
            else:
                x = actions[:, action_dim_x]
                y = actions[:, action_dim_y]
            ax.scatter(x, y, c='b', marker='*', zorder=5)

        if intention < num_intentions:
            ax.set_title('Sub-Task %02d' % (intention+1),
                         fontdict={'fontsize': 30,
                                   'fontweight': 'medium'}
                         )

    plt.show()
